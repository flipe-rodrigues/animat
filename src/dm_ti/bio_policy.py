import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.distributions import Normal, TransformedDistribution, Distribution
from torch.distributions.transforms import SigmoidTransform
from torch.distributions import constraints
from tianshou.policy import PPOPolicy

from bio_networks import LIFNeuronLayer, ModalitySpecificEncoder


# Custom distribution with entropy calculation
class SigmoidNormal(Distribution):
    """Normal distribution transformed by sigmoid for values in (0, 1)."""
    
    # Define parameter constraints
    arg_constraints = {
        'loc': constraints.real,
        'scale': constraints.positive
    }
    
    def __init__(self, loc, scale, validate_args=None):
        self.loc = loc
        self.scale = scale
        self.base_dist = Normal(loc, scale)
        self.transforms = [SigmoidTransform()]
        self.transformed_dist = TransformedDistribution(self.base_dist, self.transforms)
        
        # Initialize necessary Distribution attributes
        batch_shape = self.base_dist.batch_shape
        event_shape = self.base_dist.event_shape
        super().__init__(batch_shape, event_shape, validate_args=validate_args)
    
    def sample(self, sample_shape=torch.Size()):
        """Sample from the sigmoid-transformed normal."""
        return self.transformed_dist.sample(sample_shape)
    
    def rsample(self, sample_shape=torch.Size()):
        """Reparameterized sampling for backprop."""
        return self.transformed_dist.rsample(sample_shape)
    
    def log_prob(self, value):
        """Compute log probability, accounting for the transformation."""
        return self.transformed_dist.log_prob(value)
    
    def entropy(self):
        """Approximate entropy of the transformed distribution.
        
        This is an approximation using a Taylor series expansion
        around the mean, valid when the scale is relatively small.
        """
        # Start with the entropy of base distribution
        # This is a reasonable approximation when scale is small
        return self.base_dist.entropy()


def make_spiking_actor_critic(
    obs_dim: int,
    action_dim: int,
    hidden_size: int = 128,
    tau_mem: float = 20.0,
    tau_adapt: float = 100.0,
    adapt_scale: float = 0.05,
    lr: float = 3e-4,
    device: str = None,
    scheduler_factory=None,
) -> PPOPolicy:
    """
    Build a unified spiking actor-critic network and wrap it in a Tianshou PPOPolicy.
    Compatible with Tianshou v0.5.x API which expects separate actor and critic.
    
    Args:
        obs_dim: Observation dimension
        action_dim: Action dimension
        hidden_size: Size of hidden layer
        tau_mem: Membrane time constant
        tau_adapt: Adaptation time constant
        adapt_scale: Adaptation scaling factor
        lr: Learning rate
        device: Device to run on
        scheduler_factory: Optional function to create LR scheduler
        action_scaling: Whether to scale actions to environment bounds
        action_bound_method: Method to bound actions ('clip' or 'tanh')
    """
    device = torch.device(device) if isinstance(device, str) else device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Shared network implementation remains the same
    class SpikingActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            # shared encoder
            self.encoder = ModalitySpecificEncoder(
                target_size=12,  # Adjust target size as needed
                device=device,
            )
            enc_out = self.encoder.output_size
            
            # input projection
            self.proj = nn.Linear(enc_out, hidden_size)
            
            # Create LIFNeuronLayer directly
            self.recurrent = LIFNeuronLayer(
                size=hidden_size,
                tau_mem=tau_mem,
                tau_dist='uniform',
                tau_adapt=tau_adapt,
                adapt_scale=adapt_scale,
                device=device,
            )
            
            # policy head
            self.action_mean = nn.Linear(hidden_size, action_dim)
            self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * -1.0)
            
            # value head
            self.value_head = nn.Linear(hidden_size, 1)
            
            # Register recurrent state buffers for automatic device management
            self.register_buffer('_empty_hidden', torch.zeros(1, hidden_size))
            self.register_buffer('_empty_adapt', torch.zeros(1, hidden_size))
            
            # Move all components to device (only needed once)
            self.to(device)
            
            print(f"Unified network using device: {device}")
            print(f"Encoder output size: {enc_out}, Hidden size: {hidden_size}")
            
        def forward(self, obs, state=None, info=None):
            """Shared forward pass"""
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
                
            # State handling with registered buffers
            batch_size = x.shape[0]
            
            if state is None:
                # Create properly sized hidden state tensors from buffers
                h = self._empty_hidden.expand(batch_size, -1).clone()
                a = self._empty_adapt.expand(batch_size, -1).clone()
            else:
                # Use provided state, ensuring on correct device
                h = state.get('hidden', self._empty_hidden.expand(batch_size, -1).clone())
                a = state.get('adaptation', self._empty_adapt.expand(batch_size, -1).clone())
                
                # Ensure they're on the right device if coming from outside
                if h.device != device:
                    h = h.to(device)
                if a.device != device:
                    a = a.to(device)
                
            # Process through shared trunk
            enc = self.encoder(x)
            proj = torch.relu(self.proj(enc))
            
            # LIF recurrent layer with membrane persistence across calls
            spikes, new_adapt, new_membrane = self.recurrent(proj, a, h)
            clipped = torch.clamp(spikes, -5.0, 5.0)
            
            # Policy outputs (action distribution parameters)
            mean = torch.sigmoid(torch.clamp(self.action_mean(clipped), -5.0, 5.0))
            log_std = torch.clamp(self.action_log_std.expand(mean.size(0), -1), -2.0, 2.0)
            
            # Value output - DETACH clipped to prevent critic gradients from affecting recurrent layer
            value = self.value_head(clipped.detach()).squeeze(-1)
            
            # Return both actor and critic outputs plus state for recurrence
            return (mean, log_std), value, {
                'hidden': new_membrane.detach(),
                'adaptation': new_adapt.detach()
            }

    # Actor wrapper that satisfies the Tianshou policy interface
    class ActorWrapper(nn.Module):
        state_info_keys = ['hidden', 'adaptation']  # track recurrent state

        def __init__(self, shared_net):
            super().__init__()
            self.net = shared_net

        def forward(self, obs, state=None, info=None):
            # shared_net.forward → (mean, log_std), value, new_state
            (mean, log_std), _, new_state = self.net(obs, state, info)
            # actor returns (logits, new_state)
            return (mean, log_std), new_state

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
    shared_net = SpikingActorCritic().to(device)

    # Wrap into actor & critic
    actor = ActorWrapper(shared_net).to(device)
    critic = CriticWrapper(shared_net).to(device)

    # Optimizer over all parameters
    optim = Adam(shared_net.parameters(), lr=lr)
    
    # Create scheduler if factory provided
    lr_scheduler = None
    if scheduler_factory is not None:
        lr_scheduler = scheduler_factory(optim)

    # Distribution creation function for action sampling
    def dist_fn(mean, log_std):
        # Use our custom distribution with entropy implementation
        return SigmoidNormal(mean, log_std.exp())

    # Create and return PPO policy with separate actor/critic for compatibility
    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist_fn,
        discount_factor=0.99,
        max_grad_norm=0.5,
        eps_clip=0.2,
        vf_coef=0.5,
        ent_coef=0.02,
        reward_normalization=False,
        gae_lambda=0.95,
        action_scaling=False,  # Turn off Tianshou's action scaling
        action_bound_method='',  # No need for additional bounding
        lr_scheduler=lr_scheduler,
    )
    return policy

