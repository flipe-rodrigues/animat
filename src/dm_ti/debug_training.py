import os
import time
import torch
import numpy as np
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Union, List, Tuple

from tianshou.data import Collector, VectorReplayBuffer, Batch
from tianshou.policy import PPOPolicy
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.trainer import OnpolicyTrainer

from shimmy_wrapper import create_env
from policy_networks import RecurrentActorNetwork, CriticNetwork
from networks import ModalitySpecificEncoder
from torch.distributions import Independent, Normal


class DebugCollector(Collector):
    """Extended Collector with debugging capabilities."""
    
    def __init__(self, *args, **kwargs):
        self.debug_frequency = kwargs.pop('debug_frequency', 100)
        self.debug_dir = kwargs.pop('debug_dir', 'debug_outputs')
        super().__init__(*args, **kwargs)
        os.makedirs(self.debug_dir, exist_ok=True)
        self.episode_count = 0
        self.hidden_states_history = []
        self.action_history = []
        self.value_history = []
        self.debug_metrics = {
            'rnn_gradient_norm': [],
            'action_distribution_entropy': [],
            'hidden_state_change': [],
            'hidden_state_norm': []
        }
    
    def reset_env(self, **kwargs):
        """Override to capture state resets, with compatibility for gym_reset_kwargs."""
        if hasattr(self.policy, 'actor') and hasattr(self.policy.actor, 'net'):
            if hasattr(self.policy.actor.net, 'hidden_state_before_reset'):
                # Store current state before reset
                if self.data.state is not None and 'hidden' in self.data.state:
                    self.policy.actor.net.hidden_state_before_reset = self.data.state['hidden'].clone()
        
        # Forward all kwargs to the parent reset_env method, but don't specify env_id
        return super().reset_env(**kwargs)
    
    def collect(self, *args, **kwargs):
        """Collect experience while monitoring network behavior."""
        result = super().collect(*args, **kwargs)
        self.episode_count += 1
        
        # Debug on specified frequency
        if self.episode_count % self.debug_frequency == 0:
            self._debug_network_state()
            
        return result
    
    def _debug_network_state(self):
        """Analyze and visualize network state."""
        if not hasattr(self.policy, 'actor') or not hasattr(self.policy.actor, 'net'):
            return
            
        # Compute gradient norms if parameters have gradients
        if hasattr(self.policy.actor.net, 'recurrent'):
            rnn = self.policy.actor.net.recurrent
            grad_norm = 0.0
            for name, param in rnn.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.norm().item()
                    grad_norm += param_norm
                    print(f"    RNN {name} grad norm: {param_norm:.6f}")
            
            self.debug_metrics['rnn_gradient_norm'].append(grad_norm)
            print(f"    Total RNN gradient norm: {grad_norm:.6f}")
            
        # Visualize hidden state and action distribution
        if self.hidden_states_history and len(self.hidden_states_history) > 1:
            # Calculate hidden state change
            state_changes = []
            for i in range(1, len(self.hidden_states_history)):
                change = torch.norm(
                    self.hidden_states_history[i] - self.hidden_states_history[i-1]
                ).item()
                state_changes.append(change)
                
            self.debug_metrics['hidden_state_change'].extend(state_changes)
            
            # Plot hidden state evolution
            plt.figure(figsize=(15, 10))
            
            # Flatten all states into one 2D array
            all_states = torch.cat(self.hidden_states_history, dim=0).detach().cpu().numpy()
            
            plt.subplot(2, 2, 1)
            plt.title(f"Hidden State Evolution (Episode {self.episode_count})")
            plt.imshow(all_states, aspect='auto', cmap='viridis')
            plt.colorbar(label="Activation")
            plt.xlabel("Hidden Unit")
            plt.ylabel("Timestep")
            
            plt.subplot(2, 2, 2)
            plt.title("Hidden State Change Between Steps")
            plt.plot(state_changes)
            plt.xlabel("Step")
            plt.ylabel("L2 Norm of State Change")
            plt.grid(True)
            
            # Plot action distribution if available
            if self.action_history:
                actions = torch.cat(self.action_history, dim=0).detach().cpu().numpy()
                plt.subplot(2, 2, 3)
                plt.title("Actions Over Time")
                for i in range(actions.shape[1]):
                    plt.plot(actions[:, i], label=f"Action {i}")
                plt.legend()
                plt.xlabel("Step")
                plt.ylabel("Action Value")
                
            # Plot values if available
            if self.value_history:
                values = torch.cat(self.value_history, dim=0).detach().cpu().numpy()
                plt.subplot(2, 2, 4)
                plt.title("Value Estimates Over Time")
                plt.plot(values)
                plt.xlabel("Step")
                plt.ylabel("Value Estimate")
            
            plt.tight_layout()
            plt.savefig(f"{self.debug_dir}/episode_{self.episode_count}_debug.png")
            plt.close()
            
        # Calculate hidden state activation statistics
        if self.hidden_states_history:
            activations = torch.cat(self.hidden_states_history, dim=0)
            mean_act = activations.mean().item()
            std_act = activations.std().item()
            max_act = activations.max().item()
            min_act = activations.min().item()
            
            print(f"    Hidden state stats - Mean: {mean_act:.4f}, Std: {std_act:.4f}, "
                  f"Min: {min_act:.4f}, Max: {max_act:.4f}")
                  
            # Check for saturation (values close to -1 or 1)
            saturation = ((activations.abs() > 0.95).float().mean() * 100).item()
            print(f"    Saturation: {saturation:.2f}% of units > 0.95")
            
        # Clear histories to prevent memory issues
        self.hidden_states_history = []
        self.action_history = []
        self.value_history = []

    def track_activations(self, hidden_states, episode):
        """Track RNN activations over time"""
        if not hasattr(self, 'activation_history'):
            self.activation_history = {}
        
        # Calculate statistics
        h_mean = hidden_states.mean(dim=1).detach().cpu()  # Mean across units
        h_std = hidden_states.std(dim=1).detach().cpu()    # Std across units
        h_max = hidden_states.max(dim=1)[0].detach().cpu() # Max activation
        h_min = hidden_states.min(dim=1)[0].detach().cpu() # Min activation
        
        # Store for this episode
        self.activation_history[episode] = {
            'mean': h_mean,
            'std': h_std,
            'max': h_max,
            'min': h_min
        }


class DebugNet(nn.Module):
    """Wrapper around actor/critic networks with debugging capabilities."""
    
    def __init__(self, actor, critic, debug=True):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.debug = debug
        self.hidden_state_before_reset = None
        # For tracking over time
        self.hidden_states = []
        self.mu_history = []
        self.sigma_history = []
    
    def forward(
        self, 
        obs, 
        state=None, 
        info={}
    ) -> Tuple[Tuple, Dict]:
        """
        Forward pass with debugging capabilities.
        Returns ((mu, sigma), state).
        """
        (mu, sigma), h_n = self.actor(obs, state)
        
        # Store for debugging
        if self.debug and h_n is not None and 'hidden' in h_n:
            hidden = h_n['hidden'].detach()
            self.hidden_states.append(hidden.mean(dim=0))  # Average across batch
            self.mu_history.append(mu.detach())
            self.sigma_history.append(sigma.detach())
        
        return (mu, sigma), h_n


def make_ppo_policy(
    obs_dim: int,
    action_dim: int,
    action_space,
    hidden_size: int = 128,
    num_layers: int = 1,
    tau_mem: float = 20.0,
    lr: float = 1e-4,
    debug: bool = True,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"
) -> PPOPolicy:
    """
    Create a PPO policy with recurrent actor and critic networks.
    
    Returns:
        PPOPolicy: Policy object with debugging capabilities
    """
    print(f"Creating policy with device={device}, debug={debug}")
    
    # Create shared encoder for both actor and critic
    encoder = ModalitySpecificEncoder(target_size=40)
    
    # Create actor and critic
    actor = RecurrentActorNetwork(
        obs_shape=(obs_dim,),
        action_shape=(action_dim,),
        hidden_size=hidden_size,
        num_layers=num_layers,
        device=device,
        debug=debug,
        encoder=encoder  # Share encoder
    )
    
    critic = CriticNetwork(
        obs_shape=(obs_dim,),
        hidden_size=hidden_size,
        device=device,
        encoder=encoder  # Share encoder
    )
    
    # Wrap in debug-enabled net
    net = DebugNet(actor, critic, debug=debug)
    
    # Optimization parameters
    optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=lr
    )
    
    # Define the distribution function
    def dist_fn(logits):
        mu, sigma = logits  # Unpack the tuple returned by the actor
        # Use Independent to properly handle multi-dimensional action spaces
        return Independent(Normal(mu, sigma), 1)  # The '1' means we wrap the last dimension
    
    # Create PPO policy
    policy = PPOPolicy(
        actor=net.actor,
        critic=net.critic,
        optim=optim,
        dist_fn=dist_fn,  # Use our custom distribution function
        action_space=action_space,
        discount_factor=0.99,
        max_grad_norm=0.5,
        gae_lambda=0.95,
        vf_coef=0.5,
        ent_coef=0.01,
        eps_clip=0.2,
        value_clip=True,
        dual_clip=None,
        advantage_normalization=True,
        recompute_advantage=False,
        deterministic_eval=True,
        action_scaling=True,
        action_bound_method="clip",
        observation_space=None,
    )
    
    return policy


def log_gradients(policy, step):
    """Log gradient information for debugging."""
    print("\n=== Gradient Analysis ===")
    # Actor gradients
    actor_grad_norm = 0.0
    actor_grad_max = 0.0
    for name, param in policy.actor.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            actor_grad_norm += grad_norm
            actor_grad_max = max(actor_grad_max, grad_norm)
            if grad_norm > 1.0:  # Only log significant gradients
                print(f"Actor {name}: grad_norm={grad_norm:.4f}")
    
    # Critic gradients
    critic_grad_norm = 0.0
    critic_grad_max = 0.0
    for name, param in policy.critic.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            critic_grad_norm += grad_norm
            critic_grad_max = max(critic_grad_max, grad_norm)
            if grad_norm > 1.0:  # Only log significant gradients
                print(f"Critic {name}: grad_norm={grad_norm:.4f}")
                
    print(f"Actor grad norm: {actor_grad_norm:.4f}, max: {actor_grad_max:.4f}")
    print(f"Critic grad norm: {critic_grad_norm:.4f}, max: {critic_grad_max:.4f}")


def log_training_stats(policy, training_stats, step):
    """Log detailed training statistics."""
    # Extract key training metrics
    policy_loss = training_stats.get('policy_loss', float('nan'))
    value_loss = training_stats.get('value_loss', float('nan'))
    entropy_loss = training_stats.get('entropy_loss', float('nan'))
    kl = training_stats.get('approx_kl', float('nan'))
    
    # Log to console and potentially to loggers
    print(f"\nStep {step} Training Stats:")
    print(f"  Policy Loss: {policy_loss:.6f}")
    print(f"  Value Loss: {value_loss:.6f}")
    print(f"  Entropy Loss: {entropy_loss:.6f}")
    print(f"  KL Divergence: {kl:.6f}")


def visualize_episode(env, policy, episode_num, save_dir='episode_visualizations'):
    """Record and visualize a full episode."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Reset environment
    obs, _ = env.reset()
    state = None
    
    # Tracking data
    observations = []
    actions = []
    rewards = []
    values = []
    hidden_states = []
    
    # Run episode
    done = False
    step = 0
    total_reward = 0
    
    while not done and step < 500:  # Prevent infinite loops
        observations.append(obs.copy())
        
        # Convert to batch
        batch = Batch(obs=np.expand_dims(obs, 0), info={})
        
        # Get policy output
        with torch.no_grad():
            result = policy.forward(batch, state=state)
            action = result.act[0].cpu().numpy()
            value = policy.critic(batch.obs).item()
            state = result.state
        
        # Store hidden state
        if hasattr(policy, 'actor') and hasattr(policy.actor, 'net') and state is not None:
            if isinstance(state, dict) and 'hidden' in state:
                hidden_states.append(state['hidden'].cpu().numpy())
            elif isinstance(state, torch.Tensor):
                hidden_states.append(state.cpu().numpy())
        
        # Execute action
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store data
        actions.append(action)
        rewards.append(reward)
        values.append(value)
        
        # Update for next step
        obs = next_obs
        step += 1
        total_reward += reward
    
    # Convert to arrays
    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)
    values = np.array(values)
    
    # Generate visualization
    plt.figure(figsize=(20, 15))
    
    # Plot observations
    plt.subplot(3, 2, 1)
    plt.title(f"Observations (Episode {episode_num})")
    plt.imshow(observations, aspect='auto')
    plt.colorbar(label="Value")
    plt.xlabel("Feature")
    plt.ylabel("Timestep")
    
    # Plot actions
    plt.subplot(3, 2, 2)
    plt.title("Actions")
    for i in range(actions.shape[1]):
        plt.plot(actions[:, i], label=f"Action {i}")
    plt.legend()
    plt.xlabel("Timestep")
    plt.ylabel("Action Value")
    
    # Plot rewards
    plt.subplot(3, 2, 3)
    plt.title(f"Rewards (Total: {total_reward:.2f})")
    plt.plot(rewards, color='green')
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    
    # Plot values
    plt.subplot(3, 2, 4)
    plt.title("Value Estimates")
    plt.plot(values, color='purple')
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    
    # Plot cumulative rewards
    plt.subplot(3, 2, 5)
    plt.title("Cumulative Reward")
    plt.plot(np.cumsum(rewards), color='orange')
    plt.xlabel("Timestep")
    plt.ylabel("Cumulative Reward")
    
    # Plot hidden state evolution if available
    if hidden_states:
        plt.subplot(3, 2, 6)
        plt.title("Hidden State Evolution")
        hidden_array = np.concatenate(hidden_states, axis=0)
        plt.imshow(hidden_array, aspect='auto', cmap='viridis')
        plt.colorbar(label="Activation")
        plt.xlabel("Hidden Unit")
        plt.ylabel("Timestep")
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/episode_{episode_num}_visualization.png")
    plt.close()
    
    return total_reward


def test_memory_capacity(policy, steps=100, delay=20, save_dir=None):
    """Test if RNN can maintain memory across timesteps."""
    # Use environment's observation shape
    env = create_env()
    obs_dim = env.observation_space.shape[0]
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Create observations: signal at beginning only
    obs_seq = torch.zeros((steps, obs_dim))
    obs_seq[0, 0] = 1.0  # Signal in first timestep
    
    # Process sequence
    state = None
    outputs = []
    hiddens = []
    
    for t in range(steps):
        batch = Batch(obs=obs_seq[t:t+1], info={})
        with torch.no_grad():
            result = policy.forward(batch, state=state)
            action = result.act
            state = result.state
            outputs.append(action)
            
            # Store hidden state - handle both dict and tensor cases
            if state is not None:
                if isinstance(state, dict) and 'hidden' in state:
                    # If state is a dictionary with 'hidden' key
                    hiddens.append(state['hidden'].detach().cpu())
                elif isinstance(state, torch.Tensor):
                    # If state is directly a tensor
                    hiddens.append(state.detach().cpu())
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot input signal
    plt.subplot(3, 1, 1)
    plt.title("Input Signal")
    plt.plot(obs_seq[:, 0].numpy())
    plt.ylabel("Input Value")
    
    # Plot output actions
    plt.subplot(3, 1, 2)
    plt.title("Output Actions")
    actions = torch.cat(outputs, dim=0).numpy()
    for i in range(actions.shape[1]):
        plt.plot(actions[:, i], label=f"Action {i}")
    plt.legend()
    plt.ylabel("Action Value")
    
    # Plot hidden state evolution - FIX: Reshape hidden_array to remove batch dimension
    plt.subplot(3, 1, 3)
    plt.title("Hidden State Evolution")
    hidden_array = torch.cat(hiddens, dim=0)
    
    # Remove the batch dimension (which is causing the error)
    if len(hidden_array.shape) == 3:  # [steps, batch=1, hidden_size]
        hidden_array = hidden_array.squeeze(1)  # Convert to [steps, hidden_size]
    
    plt.imshow(hidden_array.numpy(), aspect='auto', cmap='viridis')
    plt.colorbar(label="Activation")
    plt.xlabel("Hidden Unit")
    plt.ylabel("Timestep")
    
    plt.tight_layout()
    
    # Save to specified directory or default location
    save_path = "memory_test.png"
    if save_dir:
        save_path = os.path.join(save_dir, "memory_test.png")
    plt.savefig(save_path)
    plt.close()
    
    # Calculate memory metrics
    if len(hiddens) > 1:
        # Calculate hidden state changes
        hidden_changes = []
        for i in range(1, len(hiddens)):
            change = torch.norm(
                hiddens[i] - hiddens[i-1]
            ).item()
            hidden_changes.append(change)
            
        # Calculate decay of initial signal
        initial_impact = torch.norm(hiddens[0] - hiddens[-1]).item()
        print(f"Initial signal impact (t=0 to t={steps-1}): {initial_impact:.4f}")
        
        # Calculate if there are any oscillatory patterns
        if len(hidden_changes) > 10:
            fft_vals = np.abs(np.fft.fft(hidden_changes))
            dominant_freq = np.argmax(fft_vals[1:len(fft_vals)//2]) + 1
            print(f"Dominant frequency in hidden state changes: {dominant_freq}")
    
    return outputs, hiddens


def train_with_debugging(
    # Environment parameters
    num_envs: int = 8,  # Reduced from 16
    test_num_envs: int = 2,
    
    # Training parameters
    seed: int = 42,
    buffer_size: int = 2048,  # Reduced from 4096
    hidden_size: int = 128,
    num_layers: int = 1,
    tau_mem: float = 50.0,
    lr: float = 3e-4,  # Increased from 1e-4 for faster convergence
    epoch: int = 15,    # Reduced from 300
    step_per_epoch: int = 10000,  # Reduced from 50000
    step_per_collect: int = 1000,
    repeat_per_collect: int = 4,
    batch_size: int = 64,
    
    # Debugging parameters
    debug: bool = True,
    save_interval: int = 2,     # More frequent saving
    debug_interval: int = 1,    # Debug every epoch
    test_interval: int = 1,     # Test every epoch
    log_interval: int = 1,
    
    # Logging and visualization
    logdir: str = "log",
    project: str = "animat-rnn-ppo",
    run_name: str = None,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Faster training with comprehensive debugging enabled.
    """
    print(f"Starting training with debugging (device={device}, debug={debug})")
    
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create training environments
    print("Creating training environments...")
    train_envs = SubprocVectorEnv(
        [lambda: create_env() for _ in range(num_envs)]
    )
    
    # Create test environments
    print("Creating test environments...")
    test_envs = SubprocVectorEnv(
        [lambda: create_env() for _ in range(test_num_envs)]
    )
    
    # Get environment specs
    env = create_env()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create policy
    print("Creating policy...")
    policy = make_ppo_policy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_space=env.action_space,  # Pass the action space
        hidden_size=hidden_size,
        num_layers=num_layers,
        tau_mem=tau_mem,
        lr=lr,
        debug=debug,
        device=device,
    )
    
    # Create collector with debugging
    print("Creating collectors...")
    train_collector = DebugCollector(
        policy, 
        train_envs, 
        VectorReplayBuffer(
            buffer_size, 
            len(train_envs),
            stack_num=2  # Add sequence stacking like in train_rnn.py
        ),
        debug_frequency=debug_interval,
        debug_dir=os.path.join(logdir, "debug_train"),
    )
    
    test_collector = Collector(
        policy, 
        test_envs
    )
    
    # Create logger
    print("Setting up loggers...")
    if run_name is None:
        run_name = f"rnn-ppo-{int(time.time())}"
        
    log_path = os.path.join(logdir, run_name)
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_path)
    
    # Set up WandB logger if available
    try:
        wandb_logger = WandbLogger(
            save_interval=1,
            name=run_name,
            run_id=None,
            config=dict(
                hidden_size=hidden_size,
                num_layers=num_layers,
                tau_mem=tau_mem,
                lr=lr,
                epoch=epoch,
                buffer_size=buffer_size,
                step_per_epoch=step_per_epoch,
                step_per_collect=step_per_collect,
                repeat_per_collect=repeat_per_collect,
                batch_size=batch_size,
            ),
            project=project,
        )
        logger = TensorboardLogger(writer, wandb_logger=wandb_logger)
    except:
        logger = TensorboardLogger(writer)
    
    # Before training visualization 
    print("Creating baseline episode visualization...")
    visualize_episode(
        env=env, 
        policy=policy,
        episode_num=0,
        save_dir=os.path.join(log_path, "episode_visualizations")
    )
    
    # Training loop with debugging
    print("Starting training...")
    
    def train_fn(epoch, env_step):
        """Dynamically adjust learning rate and perform other per-update operations."""
        # Original learning rate adjustment code
        if env_step <= 0.5 * step_per_epoch * epoch:
            current_lr = lr
        elif env_step <= 0.75 * step_per_epoch * epoch:
            current_lr = lr * 0.5
        else:
            current_lr = lr * 0.25
        for param_group in policy.optim.param_groups:
            param_group["lr"] = current_lr
            
        # Incorporate update_fn functionality here - log gradients periodically
        gradient_step = env_step // step_per_collect * repeat_per_collect
        if gradient_step % log_interval == 0 and debug:
            log_gradients(policy, gradient_step)
            
        # Save policy periodically (based on epoch)
        if epoch % save_interval == 0 and env_step == 0:  # Only at beginning of epoch
            torch.save(
                policy.state_dict(),
                os.path.join(log_path, f"policy_epoch_{epoch}.pth")
            )
    
    def test_fn(epoch, env_step):
        """Test function with detailed visualization and RNN analysis."""
        # Visualize an episode every debug_interval
        if epoch % debug_interval == 0:
            visualize_episode(
                env=env,
                policy=policy,
                episode_num=epoch,
                save_dir=os.path.join(log_path, "episode_visualizations")
            )
            
        # Run memory test and RNN analysis at specific early points
        if epoch in [3, 7, 14]:
            print(f"\n=== Epoch {epoch} RNN Analysis ===")
            
            # Test memory capacity
            print("Testing RNN memory capacity...")
            outputs, hiddens = test_memory_capacity(
                policy, 
                steps=100,
                save_dir=os.path.join(log_path, f"memory_test_epoch_{epoch}")
            )
            
            # Analyze RNN dynamics more deeply
            if hasattr(policy.actor, 'net') and hasattr(policy.actor.net, 'recurrent'):
                rnn = policy.actor.net.recurrent
                
                # Calculate and log weight stats
                print("\nRNN Parameter Statistics:")
                for name, param in rnn.named_parameters():
                    if 'weight' in name:
                        weight = param.data
                        print(f"  {name}: mean={weight.mean().item():.4f}, "
                              f"std={weight.std().item():.4f}, "
                              f"min={weight.min().item():.4f}, "
                              f"max={weight.max().item():.4f}")
                        
                        # For recurrent weights, check spectral radius
                        if 'weight_hh' in name:
                            try:
                                eigenvalues = torch.linalg.eigvals(weight)
                                spec_radius = torch.max(torch.abs(eigenvalues)).item()
                                print(f"  Spectral radius: {spec_radius:.4f}")
                            except:
                                print("  Could not calculate spectral radius")
    
    # Setup trainer without update_fn
    result = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=epoch,
        step_per_epoch=step_per_epoch,
        step_per_collect=step_per_collect,
        repeat_per_collect=repeat_per_collect,
        episode_per_test=test_num_envs,
        batch_size=batch_size,
        logger=logger,
        test_in_train=False,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=None,
        save_best_fn=None,
    )
    
    # Start training
    result.run()
    
    # Save final policy
    torch.save(policy.state_dict(), os.path.join(log_path, "policy_final.pth"))
    
    # Cleanup
    train_envs.close()
    test_envs.close()
    
    # Final visualization
    print("Creating final episode visualization...")
    final_reward = visualize_episode(
        env=env,
        policy=policy,
        episode_num="final",
        save_dir=os.path.join(log_path, "episode_visualizations")
    )
    
    print(f"Training complete. Final test reward: {final_reward}")
    return result, policy


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Environment
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--test-num-envs", type=int, default=2)
    
    # Training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer-size", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--tau-mem", type=float, default=20.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--step-per-epoch", type=int, default=10000)
    parser.add_argument("--step-per-collect", type=int, default=1000)
    parser.add_argument("--repeat-per-collect", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    
    # Debugging
    parser.add_argument("--debug", action="store_true", default=True)
    parser.add_argument("--save-interval", type=int, default=2)
    parser.add_argument("--debug-interval", type=int, default=1)
    parser.add_argument("--test-interval", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=1)
    
    # Logging
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--project", type=str, default="animat-rnn-ppo")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Run training with all parameters
    result, policy = train_with_debugging(**vars(args))