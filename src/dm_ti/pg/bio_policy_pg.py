import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from tianshou.policy import PGPolicy
from collections import deque

# Import the neural network components
from bio_networks import StandardRNNLayer, ModalitySpecificEncoder

from tianshou.data import Batch

class ClippedPGPolicy(PGPolicy):
    """PGPolicy with gradient clipping and additional safeguards."""
    
    def __init__(self, *args, max_grad_norm=0.5, min_std=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_grad_norm = max_grad_norm
        self.min_std = min_std
        
    def forward(self, batch, state=None, **kwargs):
        """Override forward to enforce minimum standard deviation."""
        logits, hidden = self.actor(batch.obs, state=state, info=batch.info)  # Use self.actor not self.model
        
        if isinstance(logits, tuple):
            mean, std = logits
            # Enforce minimum standard deviation
            std = torch.max(std, torch.ones_like(std) * self.min_std)
            logits = (mean, std)
            
        return Batch(logits=logits, state=hidden)
    
    def learn(self, batch, batch_size=None, repeat=1):
        """Override learn to add gradient clipping."""
        losses = []
        for _ in range(repeat):
            self.optim.zero_grad()
            result = self(batch)
            dist = self.dist_fn(*result.logits)
            
            # Safeguard against NaNs in log_prob
            try:
                log_prob = dist.log_prob(batch.act)
            except Exception as e:
                print(f"Error in log_prob calculation: {e}")
                # Emergency fallback
                self.optim.zero_grad()
                continue
            
            # Check for NaNs or extreme values
            if torch.isnan(log_prob).any() or torch.isinf(log_prob).any():
                print("WARNING: NaN or inf detected in log_prob")
                self.optim.zero_grad()
                continue
                
            # Compute returns
            returns = batch.returns.flatten()
            
            # Clip returns to prevent extreme values
            returns = torch.clamp(returns, min=-10.0, max=10.0)
            
            # Compute loss (negative because we're maximizing)
            loss = -(log_prob * returns).mean()
            
            # Clip loss
            loss = torch.clamp(loss, min=-1000.0, max=1000.0)
            
            # Backward pass with gradient clipping
            loss.backward()
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(),  # Use self.actor not self.model
                self.max_grad_norm
            )
            
            self.optim.step()
            losses.append(loss.item())
        
        # Use average loss if multiple updates were performed
        return {"loss": sum(losses) / len(losses) if losses else 0.0}
    
def make_rnn_actor(
    obs_dim: int,
    action_dim: int,
    action_space=None,
    hidden_size: int = 128,
    tau_mem: float = 20.0,
    lr: float = 3e-4,
    device: str = None,
    scheduler_factory=None,
) -> PGPolicy:
    """
    Build an RNN actor network with a biologically-inspired architecture.
    
    Args:
        obs_dim: Observation dimension
        action_dim: Action dimension
        action_space: Gym action space
        hidden_size: Size of hidden layer
        tau_mem: Membrane time constant
        lr: Learning rate
        device: Device to run on
        scheduler_factory: Optional function to create LR scheduler
    """
    device = torch.device(device) if isinstance(device, str) else device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # RNN Actor network implementation
    class RNNActor(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Observation encoder
            self.encoder = ModalitySpecificEncoder(
                target_size=40,  # Number of neurons per dimension
                device=device,
            )
            enc_out = self.encoder.output_size
            
            self.input_norm = nn.LayerNorm(enc_out)  # Normalize the encoded vector
            
            # Input projection
            self.proj = nn.Linear(enc_out, hidden_size)
            nn.init.xavier_uniform_(self.proj.weight, gain=0.5)
            nn.init.zeros_(self.proj.bias)
            
            # Recurrent layer
            self.recurrent = StandardRNNLayer(
                size=hidden_size,
                tau_mem=tau_mem,
                device=device,
                use_layer_norm=True
            )
            
            # Policy head
            self.action_mean = nn.Linear(hidden_size, action_dim)
            nn.init.xavier_uniform_(self.action_mean.weight, gain=0.1)
            nn.init.zeros_(self.action_mean.bias)
            
            # Log std with conservative initial value
            self.action_log_std = nn.Parameter(torch.zeros(1, action_dim) - 1.0)
            
            # Initial hidden state
            self.initial_hidden = nn.Parameter(torch.zeros(1, hidden_size))
            
            self.to(device)
            
        def forward(self, obs, state=None, info=None):
            # Extract observation from tianshou batch if needed
            if isinstance(obs, dict) and 'obs' in obs:
                x = obs['obs']
            else:
                x = obs
                
            # Handle input tensor conversion
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x, dtype=torch.float32, device=device)
            elif x.device != device:
                x = x.to(device)
                
            # State handling
            batch_size = x.shape[0]
            
            if state is None:
                h = self.initial_hidden.expand(batch_size, -1)
            else:
                h = state
                if h.device != device:
                    h = h.to(device)
            
            # Process through encoder
            enc = self.encoder(x)
            enc = self.input_norm(enc)
            
            # Project with ReLU
            proj = torch.relu(self.proj(enc))
            
            # Apply recurrent layer with hidden state persistence
            outputs = self.recurrent(proj, None, h)
            new_hidden = outputs  # The new hidden state is the output
            
            # Policy outputs (with safety checks)
            mean = self.action_mean(outputs)
            
            # Safety checks and clamping
            if torch.isnan(mean).any():
                mean = torch.zeros_like(mean)
            
            mean = torch.clamp(mean, -15.0, 15.0)
            
            # Get log_std and check for NaNs
            log_std = self.action_log_std
            if torch.isnan(log_std).any():
                log_std = torch.zeros_like(log_std) - 1.0
            
            # Expand and clamp log_std
            log_std = torch.clamp(log_std.expand(mean.size(0), -1), -5.0, 2.0)
            
            return (mean, log_std), new_hidden

    # Actor wrapper that satisfies the Tianshou policy interface
    class ActorWrapper(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, obs, state=None, info=None):
            try:
                # Always initialize state if None
                if state is None:
                    batch_size = obs.shape[0]
                    state = self.net.initial_hidden.expand(batch_size, -1)
                
                # Check for NaN in input state
                if torch.isnan(state).any():
                    print("WARNING: NaN detected in state - resetting")
                    state = self.net.initial_hidden.expand(state.shape[0], -1)
                    
                # Process through network
                (mean, log_std), new_state = self.net(obs, state, info)
                
                # Check for state explosion and reset if needed
                if torch.isnan(new_state).any() or torch.abs(new_state).max() > 100:
                    print(f"WARNING: Hidden state explosion: {torch.abs(new_state).max().item()}")
                    new_state = self.net.initial_hidden.expand(new_state.shape[0], -1)
                
                # Safety checks and processing
                if torch.isnan(mean).any() or torch.isnan(log_std).any():
                    mean = torch.zeros_like(mean) if torch.isnan(mean).any() else mean
                    log_std = torch.ones_like(log_std) * -1.0 if torch.isnan(log_std).any() else log_std
                    print("WARNING: Fixed NaNs in policy parameters")
                
                # Set safer bounds
                mean = torch.clamp(mean, -8.0, 8.0)
                log_std = torch.clamp(log_std, -3.0, 0.0)  # Minimum std ~0.05, maximum 1.0
                std = torch.exp(log_std).clamp(min=0.05, max=5.0)
                
                # Always return detached state
                return (mean, std), new_state.detach()
            
            except Exception as e:
                print(f"ERROR in policy forward: {e}")
                # Emergency fallback
                if isinstance(obs, np.ndarray):
                    zeros = torch.zeros((obs.shape[0], action_dim), device=device)
                    ones = torch.ones((obs.shape[0], action_dim), device=device) * 0.1
                else:
                    zeros = torch.zeros_like(obs[:, :action_dim]) 
                    ones = torch.ones_like(zeros) * 0.1
                return (zeros, ones), self.net.initial_hidden.expand(obs.shape[0], -1).detach()

        def reset_state(self, batch_size=None):
            """Reset hidden states at episode boundaries."""
            if batch_size is None:
                return None
            return self.net.initial_hidden.expand(batch_size, -1)

    # Create network and wrap into actor
    net = RNNActor().to(device)
    actor = ActorWrapper(net).to(device)

    # Create optimizer
    optim = Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    
    # Create scheduler if factory provided
    lr_scheduler = None
    if scheduler_factory is not None:
        lr_scheduler = scheduler_factory(optim)

    # Create and return PG policy
    policy = ClippedPGPolicy(
        actor=actor,  # Add this line - model and actor are the same in PG
        optim=optim,
        dist_fn=torch.distributions.Normal,
        discount_factor=0.99,
        action_space=action_space,
        reward_normalization=True,
        action_scaling=True,
        action_bound_method='clip',
        lr_scheduler=lr_scheduler,
        max_grad_norm=0.5,
        min_std=0.05
    )
    return policy