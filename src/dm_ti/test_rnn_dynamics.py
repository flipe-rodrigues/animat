import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tianshou.data import Batch

from bio_policy_rnn import make_rnn_actor_critic
from shimmy_wrapper import create_env

# Create output directory for plots
os.makedirs("rnn_test_plots", exist_ok=True)

def test_hidden_state_persistence():
    """Test if hidden states persist and change appropriately across timesteps."""
    print("\nTesting hidden state persistence...")
    
    # Create policy with small network for testing
    policy = make_rnn_actor_critic(
        obs_dim=15,  # Match arm environment
        action_dim=4,  # 4 muscle activations
        hidden_size=32,  # Small for testing
        tau_mem=8.0,   # Match training settings
        device="cpu",
    )
    
    # Extract actor for direct testing
    actor = policy.actor
    
    # Create a sequence of observations (constant for simplicity)
    obs_seq = torch.ones((10, 15)) * 0.5
    
    # Track hidden states
    hidden_states = []
    state = None
    
    print("Processing sequence and tracking hidden states...")
    for t in range(10):
        # Process observation
        _, state = actor(obs_seq[t:t+1], state)
        
        # Store hidden state
        h_t = state['hidden'].clone()
        hidden_states.append(h_t.detach())
        
        # Print some stats
        h_norm = torch.norm(h_t).item()
        print(f"  Step {t}, Hidden state norm: {h_norm:.4f}")
    
    # Check if hidden states are changing
    hidden_changes = []
    for t in range(1, 10):
        diff = torch.norm(hidden_states[t] - hidden_states[t-1]).item()
        hidden_changes.append(diff)
        print(f"  Step {t}, Change from previous: {diff:.4f}")
    
    # Plot hidden state evolution
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Hidden State Values Over Time")
    hidden_array = torch.cat(hidden_states, dim=0).numpy()
    plt.imshow(hidden_array, aspect='auto', cmap='viridis')
    plt.colorbar(label='Activation')
    plt.xlabel('Neuron Index')
    plt.ylabel('Timestep')
    
    plt.subplot(1, 2, 2)
    plt.title("Hidden State Change Between Steps")
    plt.plot(range(1, 10), hidden_changes, marker='o')
    plt.xlabel('Timestep')
    plt.ylabel('L2 Norm of State Change')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("rnn_test_plots/hidden_state_persistence.png")
    plt.close()
    
    return hidden_states

def test_gradient_flow():
    """Test if gradients flow through time in the RNN."""
    print("\nTesting gradient flow through time...")
    
    # Create policy with small network for testing
    policy = make_rnn_actor_critic(
        obs_dim=15, 
        action_dim=4,
        hidden_size=32,
        tau_mem=8.0,
        device="cpu",
    )
    
    # Extract shared network for testing
    shared_net = policy.actor.net
    
    # Create a sequence of observations
    seq_length = 5
    obs_seq = torch.rand((seq_length, 15), requires_grad=False)
    
    # Enable gradient tracking
    shared_net.zero_grad()
    
    # Process sequence
    state = None
    outputs = []
    
    for t in range(seq_length):
        # Process observation
        (mean, _), value, new_state = shared_net(obs_seq[t:t+1], state)
        outputs.append(mean)
        state = new_state
    
    # Create loss based on final output (any arbitrary loss will do)
    loss = outputs[-1].mean()
    
    # Backpropagate
    loss.backward()
    
    # Check gradients on recurrent weights
    if hasattr(shared_net.recurrent, 'weight_hh'):
        grad_norm = shared_net.recurrent.weight_hh.grad.norm().item()
        print(f"  Recurrent weight gradient norm: {grad_norm:.6f}")
        has_grad = grad_norm > 0
    else:
        print("  No weight_hh attribute found, checking all parameters...")
        has_grad = False
        for name, param in shared_net.recurrent.named_parameters():
            if param.grad is not None and param.grad.norm() > 0:
                has_grad = True
                print(f"  {name} gradient norm: {param.grad.norm().item():.6f}")
    
    print(f"  RNN has gradients flowing through time: {has_grad}")
    return has_grad

def test_temporal_sensitivity():
    """Test if the network's output depends on the history of inputs."""
    print("\nTesting temporal sensitivity...")
    
    policy = make_rnn_actor_critic(
        obs_dim=15,
        action_dim=4,
        hidden_size=32,
        tau_mem=8.0,
        device="cpu",
    )
    
    # Create two sequences with same final observation but different history
    seq_length = 5
    
    # Sequence A: All zeros then final observation
    seq_A = torch.zeros((seq_length, 15))
    seq_A[-1] = torch.ones(15) * 0.5
    
    # Sequence B: All ones then same final observation
    seq_B = torch.ones((seq_length, 15))
    seq_B[-1] = torch.ones(15) * 0.5
    
    print("Processing two sequences with different history but identical final observation...")
    
    # Process sequence A
    state_A = None
    for t in range(seq_length):
        batch = Batch(obs=seq_A[t:t+1], info={})
        result = policy.forward(batch, state=state_A)
        state_A = result.state
    output_A = result.act[0]
    
    # Process sequence B
    state_B = None
    for t in range(seq_length):
        batch = Batch(obs=seq_B[t:t+1], info={})
        result = policy.forward(batch, state=state_B)
        state_B = result.state
    output_B = result.act[0]
    
    # Compare outputs
    diff_norm = torch.norm(output_A - output_B).item()
    print(f"  Output difference (L2 norm): {diff_norm:.6f}")
    
    # Simple visualization
    plt.figure(figsize=(10, 5))
    plt.title("Action Outputs for Different Histories")
    plt.bar(range(4), output_A.detach().numpy(), alpha=0.7, label='Seq A')
    plt.bar(range(4), output_B.detach().numpy(), alpha=0.7, label='Seq B')
    plt.xlabel('Action Component')
    plt.ylabel('Action Value')
    plt.legend()
    plt.savefig("rnn_test_plots/temporal_sensitivity.png")
    plt.close()
    
    return diff_norm > 1e-4  # True if network is temporally sensitive

def test_with_memory_task():
    """A simple memory task: remember first observation bit after delay."""
    print("\nTesting with memory task...")
    
    class MemoryTaskEnv:
        """Simple synthetic environment for memory task testing."""
        def __init__(self):
            self.target = 0
            self.step_count = 0
            self.max_steps = 5
            
        def reset(self):
            self.step_count = 0
            self.target = np.random.choice([0, 1])
            obs = np.zeros(15)
            obs[0] = self.target  # Put target in first position
            return obs, {}
            
        def step(self, action):
            self.step_count += 1
            terminated = self.step_count >= self.max_steps
            
            # Only first observation has signal, rest are zeros
            obs = np.zeros(15)
            
            # At final step, reward depends on matching initial bit
            if terminated:
                # Action[0] should match target (first signal)
                reward = 1.0 if abs(action[0] - self.target) < 0.3 else 0.0
            else:
                reward = 0.0
                
            return obs, reward, terminated, False, {'target': self.target}
    
    # Create environment and policy
    env = MemoryTaskEnv()
    
    policy = make_rnn_actor_critic(
        obs_dim=15,
        action_dim=4,
        hidden_size=32,
        tau_mem=40.0,  # Increased from 20.0 to improve memory persistence
        device="cpu",
    )
    
    # Train on memory task
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.03)  # Higher learning rate
    episodes = 500
    rewards = []
    
    print("  Training on memory task...")
    
    for episode in range(episodes):
        # Start with shorter memory spans, gradually increase
        env.max_steps = min(2 + episode // 100, 5)
        
        obs, _ = env.reset()
        state = None
        episode_reward = 0
        first_obs = obs.copy()  # Store first observation for later
        
        while True:
            # Properly create batch with environment ID
            batch = Batch(
                obs=np.expand_dims(obs, 0),
                info={'env_id': np.array([0])}
            )
            
            # Get action from policy
            result = policy.forward(batch, state=state)
            action = result.act[0].detach().numpy()
            state = result.state
            
            # Step environment
            obs, reward, terminated, _, info = env.step(action)
            episode_reward += reward
            
            if terminated:
                rewards.append(episode_reward)
                if episode % 10 == 0:
                    avg_reward = np.mean(rewards[-10:]) if rewards else 0
                    print(f"  Episode {episode}, Reward: {episode_reward:.1f}, Avg: {avg_reward:.2f}")
                
                # Apply supervised learning signal with gradient-preserving outputs
                optimizer.zero_grad()
                
                # Feed the first observation again for supervised learning
                batch = Batch(
                    obs=np.expand_dims(first_obs, 0),
                    info={'env_id': np.array([0])}
                )
                
                # Get raw action outputs DIRECTLY FROM THE NETWORK to preserve gradients
                mean, _ = policy.actor.net(batch.obs, None, None)[0]  # This preserves gradients
                
                # Create binary cross-entropy loss for clearer signal
                target = float(env.target)
                mean_output = mean[0, 0].unsqueeze(0)  # FIX: Add dimension to make it shape [1]
                
                # Binary cross-entropy with logits is better for 0/1 decisions
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    mean_output, 
                    torch.tensor([target], device=mean.device)
                )
                
                # Add L2 regularization to prevent overtraining
                l2_reg = 0.001 * sum(p.pow(2.0).sum() for p in policy.parameters())
                loss = loss + l2_reg
                
                loss.backward()
                optimizer.step()
                break
    
    # Evaluate final performance  
    eval_episodes = 20
    success = 0
    
    print("  Evaluating memory task performance...")
    
    for _ in range(eval_episodes):
        obs, _ = env.reset()
        state = None
        
        while True:
            batch = Batch(
                obs=np.expand_dims(obs, 0),
                info={'env_id': np.array([0])}
            )
            
            result = policy.forward(batch, state=state) 
            action = result.act[0].detach().numpy()
            state = result.state
            
            obs, reward, terminated, _, _ = env.step(action)
            
            if terminated:
                if reward > 0:
                    success += 1
                break
    
    success_rate = success / eval_episodes
    print(f"  Memory task success rate: {success_rate:.2f}")
    
    # Plot training progress if we have rewards
    if rewards:
        plt.figure()
        window_size = min(10, len(rewards))
        if window_size > 1:
            smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(smoothed)
        else:
            plt.plot(rewards)
            
        plt.title("Memory Task Learning Progress")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.savefig("rnn_test_plots/memory_task.png")
        plt.close()
    
    return success_rate

def test_with_environment():
    """Test RNN behavior with the actual arm environment."""
    print("\nTesting with arm environment...")
    
    env = create_env()
    
    policy = make_rnn_actor_critic(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_size=32, 
        tau_mem=8.0,
        device="cpu",
    )
    
    # Run episode while tracking hidden states
    obs, _ = env.reset()
    state = None
    hidden_states = []
    actions = []
    
    for _ in range(100):  # Run for 100 steps
        batch = Batch(obs=np.expand_dims(obs, 0), info={})
        result = policy.forward(batch, state=state)
        action = result.act[0].numpy()
        state = result.state
        
        # Store hidden state
        hidden_states.append(state['hidden'].detach().cpu().numpy())
        actions.append(action)
        
        # Step environment
        obs, _, done, _, _ = env.step(action)
        if done:
            break
    
    # Visualize hidden state evolution
    hidden_array = np.concatenate(hidden_states, axis=0)
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.title("Hidden State Evolution")
    plt.imshow(hidden_array, aspect='auto', cmap='viridis')
    plt.colorbar(label="Activation")
    plt.xlabel("Neuron")
    plt.ylabel("Timestep")
    
    plt.subplot(2, 1, 2)
    plt.title("Action Trajectories")
    actions_array = np.array(actions)
    for i in range(actions_array.shape[1]):
        plt.plot(actions_array[:, i], label=f"Action {i}")
    plt.xlabel("Timestep")
    plt.ylabel("Action Value")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("rnn_test_plots/environment_test.png")
    plt.close()
    
    print(f"  Completed {len(actions)} timesteps")
    return hidden_states, actions

def run_all_tests():
    """Run all RNN dynamics tests."""
    print("=== Running RNN Dynamics Tests ===")
    
    hidden_states = test_hidden_state_persistence()
    has_gradients = test_gradient_flow()
    is_temporal = test_temporal_sensitivity()
    memory_rate = test_with_memory_task() 
    env_results = test_with_environment()
    
    print("\n=== Test Results Summary ===")
    print(f"✓ Hidden State Persistence: Hidden states maintained across {len(hidden_states)} steps")
    print(f"✓ Gradient Flow Through Time: {has_gradients}")
    print(f"✓ Temporal Sensitivity: {is_temporal}")
    print(f"✓ Memory Task Performance: {memory_rate:.2f} success rate")
    print(f"✓ Environment Interaction: Completed with {len(env_results[1])} timesteps")
    print("\nPlots saved in 'rnn_test_plots/' directory")

if __name__ == "__main__":
    run_all_tests()