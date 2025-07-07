import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.distributions import Normal
from tianshou.policy import SACPolicy
from tianshou.data import VectorReplayBuffer

from bio_networks import ModalitySpecificEncoder, StandardRNNLayer

class RSACShared(nn.Module):
    """Shared backbone for both actor and critic"""
    
    def __init__(self, obs_dim, hidden_size=64, tau_mem=8.0, device='cpu'):
        super().__init__()
        self.device = device
        
        # 1. Encoder
        self.encoder = ModalitySpecificEncoder(
                target_size=40,  # Number of neurons per dimension
                device=device,
            )
        
        # 2. Projection layer
        self.proj = nn.Linear(self.encoder.output_size, hidden_size)
        
        # 3. RNN layer
        self.rnn = StandardRNNLayer(
            size=hidden_size,
            tau_mem=tau_mem,
            device=device
        )
        
        # 4. Initial hidden state
        self.h0 = nn.Parameter(torch.zeros(1, hidden_size))
        
        self.to(device)
    
    def forward(self, obs, state=None):
        """Forward pass through shared backbone"""
        # Process inputs
        x = obs.obs if hasattr(obs, 'obs') else obs
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        
        # Get batch size
        batch_size = x.shape[0]
        
        # Get hidden state
        if state is None:
            h = self.h0.expand(batch_size, -1)
        else:
            h = state
        
        # Run encoder and RNN
        enc = self.encoder(x)
        proj = torch.relu(self.proj(enc))
        
        # Update: Use single return value from RNN as both features and new hidden state
        h_new = self.rnn(proj, None, h)
        if isinstance(h_new, tuple):
            h_new = h_new[0]  # Extract first element if tuple
        features = h_new
        
        return features, h_new
    
    def reset_hidden(self, batch_size=None):
        """Reset hidden state"""
        if batch_size is None:
            return None
        return self.h0.expand(batch_size, -1)


class RSACActor(nn.Module):
    """Actor network that uses the shared backbone"""
    
    def __init__(self, shared, action_dim, device='cpu'):
        super().__init__()
        self.shared = shared
        self.device = device
        self.action_dim = action_dim
        self.max_action = 1.0  # Add this line to fix the error
        
        # Actor outputs
        hidden_size = shared.rnn.size
        self.actor_mu = nn.Linear(hidden_size, action_dim)
        self.actor_std = nn.Linear(hidden_size, action_dim)
        
        self.to(device)
    
    def forward(self, obs, state=None, info=None):
        """Forward pass for actor"""
        features, new_h = self.shared(obs, state)
        
        # Remove sigmoid to allow unbounded mean output
        mu = self.actor_mu(features)  # Unbounded output
        
        # Safety checks
        if torch.isnan(mu).any():
            print("WARNING: NaN detected in mu, resetting to zeros")
            mu = torch.zeros_like(mu)
            
        std = torch.clamp(self.actor_std(features), -20, 2).exp()
        
        # Safety checks
        if torch.isnan(std).any():
            print("WARNING: NaN detected in std, resetting to 0.1")
            std = torch.ones_like(std)
        
        return (mu, std), new_h
    
    def reset_hidden(self, batch_size=None):
        """Reset hidden state by delegating to shared network"""
        return self.shared.reset_hidden(batch_size)


class RSACCritic1(nn.Module):
    """Critic network 1 that uses the shared backbone"""
    
    def __init__(self, shared, action_dim, device='cpu'):
        super().__init__()
        self.shared = shared
        self.device = device
        
        # Critic components
        hidden_size = shared.rnn.size
        self.critic_combine = nn.Linear(hidden_size + action_dim, hidden_size)
        self.q = nn.Linear(hidden_size, 1)
        
        # Store last hidden state
        self.last_state = None
        
        self.to(device)
    
    def forward(self, obs, act, state=None, info=None):
        """Forward pass for critic with proper batch handling"""
        # Use provided state or internal state
        if state is None and self.last_state is not None:
            state = self.last_state
        
        # Get batch size from observation
        batch_size = obs.shape[0] if hasattr(obs, 'shape') else obs.obs.shape[0]
        
        # Ensure state has correct batch dimension
        if state is not None and state.size(0) != batch_size:
            # If state doesn't match batch size, use None to trigger reset
            state = None
            
        features, new_h = self.shared(obs, state)
        
        # Detach features to prevent backprop into shared network
        features = features.detach()
        
        # Ensure action is a tensor on the right device
        if not isinstance(act, torch.Tensor):
            act = torch.tensor(act, dtype=torch.float32, device=self.device)
        elif act.device != self.device:
            act = act.to(self.device)
            
        # More explicit reshaping logic
        if act.ndim == 3:
            # Flatten all dimensions after batch
            act = act.reshape(batch_size, -1)
        
        if features.ndim == 3:
            # Flatten RNN sequence dimension if present
            features = features.reshape(batch_size, -1)
            
        # Ensure consistent dimensions
        assert features.shape[0] == batch_size, f"Features batch dimension mismatch: {features.shape}"
        assert act.shape[0] == batch_size, f"Actions batch dimension mismatch: {act.shape}"
        
        # Combine features with action
        combined = torch.cat([features, act], dim=1)
        critic_features = torch.relu(self.critic_combine(combined))
        
        # Get Q-value but DON'T squeeze it
        q = self.q(critic_features)  # This is [batch_size, 1]
        
        # Store state for next call
        self.last_state = new_h.detach()  # Detach to prevent gradient leakage
        
        # Return Q-value WITH the extra dimension
        return q
        
    def get_hidden(self):
        """Get the last hidden state"""
        return self.last_state
    
    def reset_hidden(self, batch_size=None):
        """Reset hidden state by delegating to shared network"""
        self.last_state = None
        return self.shared.reset_hidden(batch_size)


class RSACCritic2(nn.Module):
    """Critic network 2 that uses the shared backbone"""
    
    def __init__(self, shared, action_dim, device='cpu'):
        super().__init__()
        self.shared = shared
        self.device = device
        
        # Critic components
        hidden_size = shared.rnn.size
        self.critic_combine = nn.Linear(hidden_size + action_dim, hidden_size)
        self.q = nn.Linear(hidden_size, 1)
        
        # Store last hidden state
        self.last_state = None
        
        self.to(device)
    
    def forward(self, obs, act, state=None, info=None):
        """Forward pass for critic with proper batch handling"""
        # Use provided state or internal state
        if state is None and self.last_state is not None:
            state = self.last_state
        
        # Get batch size from observation
        batch_size = obs.shape[0] if hasattr(obs, 'shape') else obs.obs.shape[0]
        
        # Ensure state has correct batch dimension
        if state is not None and state.size(0) != batch_size:
            # If state doesn't match batch size, use None to trigger reset
            state = None
            
        features, new_h = self.shared(obs, state)
        
        # Detach features to prevent backprop into shared network
        features = features.detach()
        
        # Ensure action is a tensor on the right device
        if not isinstance(act, torch.Tensor):
            act = torch.tensor(act, dtype=torch.float32, device=self.device)
        elif act.device != self.device:
            act = act.to(self.device)
            
        # More explicit reshaping logic
        if act.ndim == 3:
            # Flatten all dimensions after batch
            act = act.reshape(batch_size, -1)
        
        if features.ndim == 3:
            # Flatten RNN sequence dimension if present
            features = features.reshape(batch_size, -1)
            
        # Ensure consistent dimensions
        assert features.shape[0] == batch_size, f"Features batch dimension mismatch: {features.shape}"
        assert act.shape[0] == batch_size, f"Actions batch dimension mismatch: {act.shape}"
        
        # Combine features with action
        combined = torch.cat([features, act], dim=1)
        critic_features = torch.relu(self.critic_combine(combined))
        
        # Get Q-value but DON'T squeeze it
        q = self.q(critic_features)  # This is [batch_size, 1]
        
        # Store state for next call
        self.last_state = new_h.detach()  # Detach to prevent gradient leakage
        
        # Return Q-value WITH the extra dimension
        return q
        
    def get_hidden(self):
        """Get the last hidden state"""
        return self.last_state
    
    def reset_hidden(self, batch_size=None):
        """Reset hidden state by delegating to shared network"""
        self.last_state = None
        return self.shared.reset_hidden(batch_size)


def create_simple_sac(
    obs_dim,
    action_dim,
    action_space,
    hidden_size=64,
    tau_mem=8.0,
    lr=3e-4,
    alpha=0.2,         
    auto_alpha=False,  
    device='cpu',
    debug=False
):
    shared = RSACShared(obs_dim, hidden_size, tau_mem, device)
    actor = RSACActor(shared, action_dim, device)
    
    # Create separate critics
    critic1 = RSACCritic1(shared, action_dim, device)
    critic2 = RSACCritic2(shared, action_dim, device)

    # Fix duplicate parameter warning
    actor_params = list(dict.fromkeys(list(shared.parameters()) + list(actor.parameters())))
    actor_optim = Adam(actor_params, lr=lr)
    critic1_optim = Adam(critic1.parameters(), lr=lr)
    critic2_optim = Adam(critic2.parameters(), lr=lr)

    # Set up alpha for auto-tuning if requested
    if auto_alpha:
        target_entropy = -float(action_dim) # -dim(A)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=lr)
        alpha_param = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha_param = alpha

    # Use critics directly - no adapters needed
    policy = SACPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic=critic1,
        critic_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=0.005, 
        gamma=0.99,
        alpha=alpha_param,
        estimation_step=1,
        action_space=action_space,
        action_scaling=True,  # Keep action scaling
        action_bound_method='clip',  # Change from 'tanh' to 'clip'
        deterministic_eval=True
    )
    
    policy.reset_hidden = lambda b: shared.reset_hidden(b)
    
    # Set flag for debug mode
    policy._debug_mode = debug

    # Mark policy as recurrent and patch learn method
    policy._is_recurrent = True
    
    # Store original learn method
    original_learn = policy.learn
    
    # New learn method that handles states correctly
    def learn_with_hidden_state(batch, **kwargs):
        # Extract hidden states from batch.info if available
        hidden_states = None
        if hasattr(batch, 'info') and batch.info is not None:
            hidden_states = []
            for info in batch.info:
                if hasattr(info, 'hidden_state'):
                    hidden_states.append(info.hidden_state)
        
        # Pass hidden states to the original learn method
        return original_learn(batch, hidden_states=hidden_states, **kwargs)
    
    policy.learn = learn_with_hidden_state
    
    return policy


def create_collector(policy, env, buffer, exploration_noise=True):
    """Create a collector with proper recurrent state handling."""
    from tianshou.data import Collector
    
    # Create collector without hook parameter (it doesn't exist)
    collector = Collector(
        policy, env, buffer,
        exploration_noise=exploration_noise
    )
    
    # Monkey patch the _reset_hidden_state_based_on_type method to sync all components
    original_reset_hidden = collector._reset_hidden_state_based_on_type
    
    def enhanced_reset_hidden(env_ind_local_D, last_hidden_state_RH):
        # Call original reset first
        original_reset_hidden(env_ind_local_D, last_hidden_state_RH)
        
        # Then synchronize all hidden states for the completed episodes
        # This ensures critics get their states reset too
        sync_hidden_states(policy)
    
    collector._reset_hidden_state_based_on_type = enhanced_reset_hidden
    
    # Ensure proper reset at collector initialization
    original_reset = collector.reset
    
    def reset_with_sync(*args, **kwargs):
        result = original_reset(*args, **kwargs)
        sync_hidden_states(policy)
        return result
    
    collector.reset = reset_with_sync
    
    return collector


def sync_hidden_states(policy):
    """Synchronize hidden states across components at episode boundaries"""
    # Get fresh state from shared backbone
    batch_size = 1  # Or appropriate size for your environment
    shared_state = policy.actor.shared.reset_hidden(batch_size)
    
    # Set state for all components
    policy.critic.last_state = shared_state.detach() if shared_state is not None else None
    policy.critic2.last_state = shared_state.detach() if shared_state is not None else None
    
    return shared_state


class RecurrentReplayBuffer(VectorReplayBuffer):
    """Buffer that stores and replays hidden states for RNN policies"""
    
    def add(self, batch, buffer_ids=None):
        """Store states along with transitions"""
        # Get the policy's latest hidden state if available
        if hasattr(batch, 'info') and batch.info is not None:
            for i, info in enumerate(batch.info):
                if hasattr(info, 'hidden_state') and info.hidden_state is not None:
                    # Store hidden state in the info field
                    if buffer_ids is not None:
                        self.buffers[buffer_ids[i]].info[self._index[buffer_ids[i]]].hidden_state = info.hidden_state
                    else:
                        # Handle non-vectorized case
                        self.info[self._index].hidden_state = info.hidden_state
        
        # Call the original add method
        return super().add(batch, buffer_ids)
    
    def sample(self, batch_size):
        """Include hidden states in sampled batch"""
        batch = super().sample(batch_size)
        # Extract hidden states from the info field
        # This gets passed to the policy.learn function
        return batch

