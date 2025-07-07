import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from tianshou.policy import PPOPolicy

# Change this import to use the population encoder
from bio_networks import StandardRNNLayer, ModalitySpecificEncoder  # Updated import


def make_rnn_actor_critic(
    obs_dim: int,
    action_dim: int,
    action_space=None,  
    hidden_size: int = 128,
    tau_mem: float = 20.0,
    tau_adapt: float = None,
    adapt_scale: float = None,
    lr: float = 3e-4,
    device: str = None,
    scheduler_factory=None,
) -> PPOPolicy:
    """
    Build a unified RNN actor-critic network with grid cell encoding.
    
    Args:
        obs_dim: Observation dimension
        action_dim: Action dimension
        hidden_size: Size of hidden layer
        tau_mem: Membrane time constant
        lr: Learning rate
        device: Device to run on
        scheduler_factory: Optional function to create LR scheduler
    """
    device = torch.device(device) if isinstance(device, str) else device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Shared network implementation
    class RNNActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Replace grid cell encoder with population encoder
            self.encoder = ModalitySpecificEncoder(
                target_size=40,  # Number of neurons per dimension
                device=device,
            )
            enc_out = self.encoder.output_size
            
            self.input_norm = nn.LayerNorm(enc_out)  # Normalize the entire encoded vector
            
            # Input projection with better initialization
            self.proj = nn.Linear(enc_out, hidden_size)
            nn.init.xavier_uniform_(self.proj.weight, gain=0.5)
            nn.init.zeros_(self.proj.bias)
            
            # Recurrent layer with stabilized updates
            self.recurrent = StandardRNNLayer(
                size=hidden_size,
                tau_mem=tau_mem,
                device=device,
                use_layer_norm=False  # Keep False for better memory
            )
            
            # Make post-recurrent normalization optional
            self.use_post_norm = False  # Set to False for better memory
            if self.use_post_norm:
                self.layer_norm = nn.LayerNorm(hidden_size)
            
            # Policy head with careful initialization
            self.action_mean = nn.Linear(hidden_size, action_dim)
            nn.init.xavier_uniform_(self.action_mean.weight, gain=0.1)
            nn.init.zeros_(self.action_mean.bias)
            
            # Log std with more conservative initial value
            self.action_log_std = nn.Parameter(torch.zeros(1, action_dim) - 1.0)
            
            # Value head with careful initialization
            self.value_head = nn.Linear(hidden_size, 1)
            nn.init.xavier_uniform_(self.value_head.weight, gain=0.01)
            nn.init.zeros_(self.value_head.bias)
            
            # Better initial hidden state (small values close to zero)
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
                h = state.get('hidden')
                if h is None:
                    h = self.initial_hidden.expand(batch_size, -1)
                if h.device != device:
                    h = h.to(device)
            
            # Process through encoder (no additional normalization needed)
            enc = self.encoder(x)
            enc = self.input_norm(enc)  # Normalize the entire encoded vector
            
            # Project with ReLU
            proj = torch.relu(self.proj(enc))
            
            # Apply recurrent layer with hidden state persistence
            # FIXED: StandardRNNLayer now returns only the new hidden state
            outputs = self.recurrent(proj, None, h)
            new_hidden = outputs  # The output is the new hidden state
            
            # Apply layer norm conditionally
            if self.use_post_norm:
                outputs = self.layer_norm(outputs)
            
            # Check for NaNs in the outputs themselves
            if torch.isnan(outputs).any():
                print("WARNING: NaN detected in RNN outputs - stabilizing")
                outputs = torch.nan_to_num(outputs, nan=0.0)
            
            # Directly apply sigmoid to mean values instead of tanh + clamping
            mean = torch.tanh(self.action_mean(outputs))
            
            # Safety check to detect NaNs early
            if torch.isnan(mean).any():
                # Emergency recovery - replace with zeros and log warning
                print("WARNING: NaN detected in action mean - stabilizing network")
                mean = torch.zeros_like(mean)
            
            # Adjust log_std range for actions in [0,1] space
            log_std = self.action_log_std
            if torch.isnan(log_std).any():
                print("WARNING: NaN detected in log_std - stabilizing")
                log_std = torch.zeros_like(log_std) - 1.0  # Reset to default
            
            # Expand and clamp log_std
            log_std = torch.clamp(log_std.expand(mean.size(0), -1), -5.0, -1.5)
            
            # Value function (detach safely with copy to prevent gradient flow)
            value = self.value_head(outputs.detach()).squeeze(-1)
            
            return (mean, log_std), value, {'hidden': new_hidden}

    # Actor wrapper that satisfies the Tianshou policy interface
    class ActorWrapper(nn.Module):
        def __init__(self, shared_net):
            super().__init__()
            self.net = shared_net
            
            # Add this attribute to fix the error
            self.max_action = 1.0  # Default normalization bound
            
            self.tbptt_steps = 8  # Increased from 20
            self._state = None
            # Track history per environment id
            self.env_state_history = {}

        def forward(self, obs, state=None, info=None):
            try:
                (mean, log_std), _, new_state = self.net(obs, state, info)
                
                # Extensive NaN checks and fixes
                if torch.isnan(mean).any() or torch.isnan(log_std).any():
                    # Emergency fix for both parameters
                    mean = torch.zeros_like(mean) if torch.isnan(mean).any() else mean
                    log_std = torch.ones_like(log_std) * -1.0 if torch.isnan(log_std).any() else log_std
                    print("WARNING: Fixed NaNs in policy parameters")
                    
                # Hard clamp to ensure valid values
                mean = torch.clamp(mean, -10.0, 10.0)
                log_std = torch.clamp(log_std, -5.0, 0.0)  # More conservative range
                
                # Safe standard deviation calculation
                std = torch.exp(log_std).clamp(min=1e-6, max=10.0)
                
                # Process hidden states with fixed logic
                if hasattr(info, 'env_id') and info.env_id is not None:
                    # Handle both tensor and numpy array env_ids safely
                    if isinstance(info.env_id, torch.Tensor):
                        env_ids = info.env_id.cpu().numpy()
                    else:
                        env_ids = info.env_id
                        
                    hidden_states = []
                    
                    for i, env_id in enumerate(env_ids):
                        # Initialize history for this environment if needed
                        if env_id not in self.env_state_history:
                            self.env_state_history[env_id] = []
                        
                        # Get this environment's new hidden state
                        env_hidden = new_state['hidden'][i:i+1]
                        
                        # Add to history
                        hist = self.env_state_history[env_id]
                        hist.append(env_hidden)
                        
                        # FIXED: Drop oldest state if history is too long,
                        # but always use the newest state (detached) for next step
                        if len(hist) > self.tbptt_steps:
                            hist.pop(0)  # drop oldest
                            
                        # Always use most recent hidden state (detached)
                        hidden_states.append(env_hidden.detach())
                    
                    # Combine all hidden states back into a batch
                    detached_state = {'hidden': torch.cat(hidden_states, dim=0)}
                else:
                    # For test environments or single env case - simpler approach
                    detached_state = {'hidden': new_state['hidden'].detach()}
                    
                # Return values as before
                return (mean, std), detached_state
            except Exception as e:
                print(f"ERROR in policy forward: {e}")
                # Emergency fallback with proper return format
                if isinstance(obs, torch.Tensor):
                    batch_size = obs.shape[0]
                    zeros = torch.zeros(batch_size, action_dim, device=obs.device)
                    ones = torch.ones(batch_size, action_dim, device=obs.device) * 0.1
                else:
                    # Handle numpy arrays
                    batch_size = obs.shape[0]
                    zeros = torch.zeros(batch_size, action_dim, device=device)
                    ones = torch.ones(batch_size, action_dim, device=device) * 0.1
                
                # FIXED: Return fallback values as tuple with two values
                return (zeros, ones), {'hidden': self.net.initial_hidden.expand(batch_size, -1).detach()}

        def reset_state(self, batch_size=None, done_env_ids=None):
            """Reset hidden states only for environments that are done."""
            if done_env_ids is not None:
                # Selective reset - only clear history for finished environments
                for env_id in done_env_ids:
                    if env_id in self.env_state_history:
                        del self.env_state_history[env_id]
            else:
                # Full reset (only when no specific done_env_ids are provided)
                self.env_state_history = {}
            
            if batch_size is None:
                self._state = None
            else:
                self._state = {'hidden': self.net.initial_hidden.expand(batch_size, -1)}
            return self._state

    # Critic wrapper that satisfies the Tianshou critic interface
    class CriticWrapper(nn.Module):
        def __init__(self, shared_net):
            super().__init__()
            self.net = shared_net

        def forward(self, obs, **kwargs):
            # shared_net.forward → (mean, log_std), value, new_state
            _, value, _ = self.net(obs, None, None)
            # critic returns value only
            return value

    # Build shared spiking network once
    shared_net = RNNActorCritic().to(device)

    # Wrap into actor & critic
    actor = ActorWrapper(shared_net).to(device)
    critic = CriticWrapper(shared_net).to(device)

    # Create optimizer with properly separated parameter groups
    def create_optimizer_with_unique_params():
        # Create parameter groups with different learning rates
        actor_head_params = []
        critic_head_params = []
        shared_params = []
        
        # Collect shared parameters first
        for name, param in shared_net.named_parameters():
            if "action_mean" in name or "action_log_std" in name:
                actor_head_params.append(param)
            elif "value_head" in name:
                critic_head_params.append(param)
            else:
                shared_params.append(param)
        
        # Display group sizes for verification
        print(f"Actor-specific params: {len(actor_head_params)}")
        print(f"Critic-specific params: {len(critic_head_params)}")
        print(f"Shared network params: {len(shared_params)}")
        
        # Create optimizer with properly separated parameter groups
        return Adam([
            {'params': actor_head_params, 'lr': 2e-4},
            {'params': critic_head_params, 'lr': 2e-4},
            {'params': shared_params, 'lr': 1e-4}  # Use intermediate LR for shared params
        ], weight_decay=3e-5)

    # Use our custom function
    optim = create_optimizer_with_unique_params()
    
    # Create scheduler if factory provided
    lr_scheduler = None
    if scheduler_factory is not None:
        lr_scheduler = scheduler_factory(optim)
        # Ensure the trainer will step the scheduler appropriately
        # Tianshou PPOPolicy handles this when lr_scheduler is provided

    # Create and return PPO policy with separate actor/critic for compatibility
    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=lambda x: torch.distributions.Independent(
            torch.distributions.Normal(*x), 1
        ),
        discount_factor=0.99,
        max_grad_norm=0.5,        # Reduced from 0.5
        eps_clip=0.2,            # CHANGE: Lower clip threshold for more conservative updates
        vf_coef=0.5,
        ent_coef=0.03,          # Reduced entropy coefficient
        action_space=action_space,
        reward_normalization=False,
        gae_lambda=0.95,
        action_scaling=True,
        action_bound_method='clip',
        lr_scheduler=lr_scheduler,
        advantage_normalization=True,
        recompute_advantage=True,
        value_clip=True,
        dual_clip=3.0            # More conservative dual clipping
    )
    return policy