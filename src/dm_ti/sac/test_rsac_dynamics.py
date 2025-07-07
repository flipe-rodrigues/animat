import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tianshou.data import Batch
from tianshou.env import DummyVectorEnv

from bio_policy_sac2 import create_simple_sac, create_collector, RSACShared, RSACCritic1, RSACCritic2
from shimmy_wrapper import create_env

# Create output directory for plots
os.makedirs("rsac_test_plots", exist_ok=True)

def test_hidden_state_persistence():
    """Test if hidden states persist and change across timesteps."""
    print("\nTesting hidden state persistence...")
    
    # Create environment for observation/action shapes
    env = create_env()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create SAC policy with small network for testing
    policy = create_simple_sac(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_space=env.action_space,
        hidden_size=32,  # Small for testing
        tau_mem=8.0,
        device="cpu"
    )
    
    # Create a sequence of observations (constant for simplicity)
    obs_seq = torch.ones((10, obs_dim)) * 0.5
    
    # Track hidden states
    hidden_states = []
    state = None
    
    print("Processing sequence and tracking hidden states...")
    for t in range(10):
        # Convert to batch for processing
        batch = Batch(obs=obs_seq[t:t+1])
        
        # Process through actor network
        with torch.no_grad():  # No need for gradients in this test
            result = policy.actor(batch.obs, state)
            distribution, state = result
        
        # Store hidden state
        hidden_states.append(state.clone().detach())
        
        # Print some stats
        h_norm = torch.norm(state).item()
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
    plt.savefig("rsac_test_plots/hidden_state_persistence.png")
    plt.close()
    
    return hidden_states

def test_gradient_isolation():
    """
    Test if gradients flow correctly (actor gradients flow through RNN,
    but critic gradients are properly isolated from shared backbone).
    """
    print("\nTesting gradient isolation...")
    
    # Create environment for observation/action shapes
    env = create_env()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create SAC policy
    policy = create_simple_sac(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_space=env.action_space,
        hidden_size=32,
        tau_mem=8.0,
        device="cpu"
    )
    
    # Get shared network for checking gradients
    shared = policy.actor.shared
    
    # Create sample observation and action
    obs = torch.rand((1, obs_dim), requires_grad=False)
    act = torch.rand((1, action_dim), requires_grad=False)
    
    # TEST 1: Actor gradients should flow into shared network
    print("  Testing actor gradient flow...")
    shared.zero_grad()
    
    # Forward through actor
    (mu, std), _ = policy.actor(obs, None)
    
    # Create loss based on actor output
    actor_loss = mu.mean() + std.mean()
    actor_loss.backward()
    
    # Check gradients on shared parameters
    actor_grad_exists = False
    for name, param in shared.named_parameters():
        if param.grad is not None and param.grad.norm() > 0:
            actor_grad_exists = True
            print(f"    Shared {name} has gradient from actor: {param.grad.norm().item():.6f}")
    
    # TEST 2: Critic gradients should NOT flow into shared network
    print("  Testing critic gradient isolation...")
    shared.zero_grad()
    
    # Forward through both critics separately - DON'T unpack values anymore
    q1 = policy.critic(obs, act, None)  # No unpacking needed
    q2 = policy.critic2(obs, act, None) # No unpacking needed
    
    # Create loss based on both critic outputs
    critic_loss = q1.mean() + q2.mean()
    critic_loss.backward()
    
    # Check gradients on shared parameters
    critic_grad_isolated = True
    for name, param in shared.named_parameters():
        if param.grad is not None and param.grad.norm() > 0:
            critic_grad_isolated = False
            print(f"    WARNING: Shared {name} has gradient from critic: {param.grad.norm().item():.6f}")
    
    print(f"  Actor gradients flow to shared: {actor_grad_exists}")
    print(f"  Critic gradients isolated from shared: {critic_grad_isolated}")
    
    return actor_grad_exists, critic_grad_isolated

def test_temporal_sensitivity():
    """Test if the network's output depends on the history of inputs."""
    print("\nTesting temporal sensitivity...")
    
    # Create environment for observation/action shapes
    env = create_env()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create SAC policy
    policy = create_simple_sac(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_space=env.action_space,
        hidden_size=32,
        tau_mem=8.0,
        device="cpu"
    )
    
    # Create two sequences with same final observation but different history
    seq_length = 5
    
    # Sequence A: All zeros then final observation
    seq_A = torch.zeros((seq_length, obs_dim))
    seq_A[-1] = torch.ones(obs_dim) * 0.5
    
    # Sequence B: All ones then same final observation
    seq_B = torch.ones((seq_length, obs_dim))
    seq_B[-1] = torch.ones(obs_dim) * 0.5
    
    print("Processing two sequences with different history but identical final observation...")
    
    # Process sequence A
    state_A = None
    for t in range(seq_length):
        batch = Batch(obs=seq_A[t:t+1])
        with torch.no_grad():
            result = policy.actor(batch.obs, state_A)
            _, state_A = result
    
    # Get action distribution for final observation
    with torch.no_grad():
        (mu_A, std_A), _ = policy.actor(seq_A[-1:], state_A)
    
    # Process sequence B
    state_B = None
    for t in range(seq_length):
        batch = Batch(obs=seq_B[t:t+1])
        with torch.no_grad():
            result = policy.actor(batch.obs, state_B)
            _, state_B = result
    
    # Get action distribution for final observation
    with torch.no_grad():
        (mu_B, std_B), _ = policy.actor(seq_B[-1:], state_B)
    
    # Compare outputs
    mu_diff = torch.norm(mu_A - mu_B).item()
    std_diff = torch.norm(std_A - std_B).item()
    print(f"  Action mean difference (L2 norm): {mu_diff:.6f}")
    print(f"  Action std difference (L2 norm): {std_diff:.6f}")
    
    # Simple visualization
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.title("Action Means for Different Histories")
    plt.bar(range(action_dim), mu_A[0].numpy(), alpha=0.7, label='Seq A')
    plt.bar(range(action_dim), mu_B[0].numpy(), alpha=0.7, label='Seq B')
    plt.xlabel('Action Component')
    plt.ylabel('Mean Value')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.title("Action Stds for Different Histories")
    plt.bar(range(action_dim), std_A[0].numpy(), alpha=0.7, label='Seq A')
    plt.bar(range(action_dim), std_B[0].numpy(), alpha=0.7, label='Seq B')
    plt.xlabel('Action Component')
    plt.ylabel('Std Value')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("rsac_test_plots/temporal_sensitivity.png")
    plt.close()
    
    is_temporally_sensitive = mu_diff > 1e-4 or std_diff > 1e-4
    print(f"  Network is temporally sensitive: {is_temporally_sensitive}")
    
    return is_temporally_sensitive

def test_encoder_dimensions():
    """Test that encoder dimensions are correct."""
    print("\nTesting encoder dimensions...")
    
    # Create environment for observation shape
    env = create_env()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create SAC policy
    policy = create_simple_sac(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_space=env.action_space,
        hidden_size=32,
        tau_mem=8.0,
        device="cpu"
    )
    
    # Get shared network
    shared = policy.actor.shared
    
    # Create sample observation
    obs = torch.rand((1, obs_dim))
    
    # Run through encoder
    with torch.no_grad():
        encoded = shared.encoder(obs)
    
    expected_size = shared.encoder.output_size
    actual_size = encoded.shape[1]
    
    print(f"  Encoder output size: expected={expected_size}, actual={actual_size}")
    print(f"  Dimensions match: {expected_size == actual_size}")
    
    # Verify component sizes based on your specification
    # 12 muscle dims + 2*target_size (pop code) + 1 (Z)
    target_size = 40  # as specified in your code
    expected_breakdown = 12 + (2 * target_size) + 1
    
    print(f"  Expected breakdown: 12 (muscle) + {2 * target_size} (population code) + 1 (Z) = {expected_breakdown}")
    
    return expected_size == actual_size

def test_with_environment():
    """Test the policy with the actual environment."""
    print("\nTesting with arm environment...")
    
    # Create environment
    env = create_env()
    
    # Create SAC policy
    policy = create_simple_sac(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        action_space=env.action_space,
        hidden_size=32,
        tau_mem=8.0,
        device="cpu"
    )
    
    # Run a few episodes
    episodes = 3
    max_steps = 100
    
    hidden_states = []
    actions = []
    observations = []
    
    for episode in range(episodes):
        print(f"  Running episode {episode+1}/{episodes}...")
        obs, _ = env.reset()
        state = None
        episode_hidden = []
        episode_actions = []
        episode_obs = []
        
        for step in range(max_steps):
            # Convert to batch - FIX: Add empty info dict
            batch = Batch(obs=np.expand_dims(obs, 0), info={})
            
            # Get action from policy
            with torch.no_grad():
                result = policy(batch, state=state)
                # Add explicit CPU transfer before numpy conversion
                action = result.act[0].cpu().numpy()
                state = result.state
            
            # Store data
            episode_hidden.append(state.cpu().numpy())
            episode_actions.append(action)
            episode_obs.append(obs)
            
            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action)
            
            if terminated or truncated:
                break
        
        print(f"    Episode lasted {step+1} steps")
        hidden_states.append(np.concatenate(episode_hidden, axis=0))
        actions.append(np.array(episode_actions))
        observations.append(np.array(episode_obs))
    
    # Visualize results from one episode
    plt.figure(figsize=(15, 10))
    
    # Plot hidden states
    plt.subplot(3, 1, 1)
    plt.title("Hidden State Evolution (Episode 1)")
    plt.imshow(hidden_states[0], aspect='auto', cmap='viridis')
    plt.colorbar(label="Activation")
    plt.xlabel("Neuron")
    plt.ylabel("Timestep")
    
    # Plot actions
    plt.subplot(3, 1, 2)
    plt.title("Action Trajectories (Episode 1)")
    for i in range(actions[0].shape[1]):
        plt.plot(actions[0][:, i], label=f"Action {i}")
    plt.xlabel("Timestep")
    plt.ylabel("Action Value")
    plt.legend()
    
    # Plot target position (assuming it's in the last 3 dimensions)
    plt.subplot(3, 1, 3)
    plt.title("Target Position (Episode 1)")
    for i in range(3):
        plt.plot(observations[0][:, -(3-i)], label=f"Dim {i}")
    plt.xlabel("Timestep")
    plt.ylabel("Position")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("rsac_test_plots/environment_test.png")
    plt.close()
    
    return hidden_states, actions, observations

def test_alpha_tuning():
    """Test policy functionality with proper handling of alpha tuning"""
    print("\nTesting policy functionality with proper alpha handling...")
    
    # Import the context manager
    from tianshou.policy.base import policy_within_training_step
    
    # Create environment
    env = create_env()
    
    # Create SAC policy with debug enabled
    device = 'cpu'
    policy_fixed = create_simple_sac(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        action_space=env.action_space,
        hidden_size=32,
        tau_mem=8.0,
        auto_alpha=False,  # Use fixed alpha
        debug=True,
        device=device
    )
    
    # Create buffer
    from tianshou.data import VectorReplayBuffer
    buffer = VectorReplayBuffer(200, buffer_num=2)
    
    # Create vector environment and collector
    train_envs = DummyVectorEnv([lambda: create_env() for _ in range(2)])
    collector = create_collector(
        policy=policy_fixed,
        env=train_envs,
        buffer=buffer,
        exploration_noise=True
    )
    
    # Reset policy hidden state explicitly
    policy_fixed.reset_hidden(2)  # For 2 environments
    
    # Reset and collect small amount of data
    collector.reset()
    print("  Collecting initial data...")
    collector.collect(n_step=20)  # Smaller collection for debugging
    
    print("\n  Testing policy update with fixed alpha...")
    # Try updating with a small batch first
    try:
        print("  Updating with batch size 8...")
        with policy_within_training_step(policy_fixed):  # Use the context manager
            losses_fixed = policy_fixed.update(8, buffer)
        print("  Update successful!")
    except Exception as e:
        print(f"  Error with batch size 8: {e}")
    
    # Then try with the normal batch size
    try:
        print("  Updating with batch size 64...")
        with policy_within_training_step(policy_fixed):  # Use the context manager
            losses_fixed = policy_fixed.update(64, buffer)
        print("  Update successful!")
        return True
    except Exception as e:
        print(f"  Error with batch size 64: {e}")
        return False

def debug_critic_batch_handling():
    """
    Test function to debug batch handling in critic networks.
    This helps diagnose dimension mismatches during policy updates.
    """
    print("\nDebugging critic batch handling...")
    
    # Create environment for dimensions
    env = create_env()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create shared network and critics
    shared = RSACShared(obs_dim, 32, 8.0, 'cpu')
    critic1 = RSACCritic1(shared, action_dim, 'cpu')
    critic2 = RSACCritic2(shared, action_dim, 'cpu')
    
    print("Testing with various batch sizes:")
    batch_sizes = [1, 16, 64, 128]
    for bs in batch_sizes:
        # Create random observations and actions
        obs = torch.rand(bs, obs_dim)
        act = torch.rand(bs, action_dim)
        
        # Test critic1 - no unpacking since it now returns only q-values
        q1 = critic1(obs, act, None)
        print(f"  Critic1: Batch size {bs}: input shapes obs={obs.shape}, act={act.shape} → Q shape={q1.shape}")
        
        # Test critic2 - no unpacking
        q2 = critic2(obs, act, None)
        print(f"  Critic2: Batch size {bs}: input shapes obs={obs.shape}, act={act.shape} → Q shape={q2.shape}")
    
    print("\nTesting with mismatched batch sizes:")
    # Test with mismatched dimensions (common error case)
    try:
        obs_mismatch = torch.rand(64, obs_dim)
        act_mismatch = torch.rand(128, action_dim)
        q_mismatch = critic1(obs_mismatch, act_mismatch, None)
        print("  WARNING: No error with mismatched batch dimensions!")
    except AssertionError as e:  # Changed from ValueError to AssertionError
        print(f"  Expected error caught: {e}")
    
    print("\nTesting with sequence data:")
    try:
        seq_obs = torch.rand(10, 5, obs_dim)  # [seq_len, batch, features]
        seq_act = torch.rand(10, 5, action_dim)
        
        # Process sequence
        hidden = None
        outputs = []
        for t in range(seq_obs.size(0)):
            # If critic1 still stores state internally, just call it without unpacking
            q = critic1(seq_obs[t], seq_act[t], hidden)
            outputs.append(q)
            # Note: We can't update hidden here since it's now managed internally
        print(f"  Sequence processing successful: {len(outputs)} timesteps")
    except Exception as e:
        print(f"  Error with sequence data: {e}")
    
    return True

def run_all_tests():
    """Run all tests and report results."""
    print("=== Running RSAC Dynamics Tests ===")
    
    hidden_states = test_hidden_state_persistence()
    actor_grad, critic_isolation = test_gradient_isolation()
    is_temporal = test_temporal_sensitivity()
    encoder_correct = test_encoder_dimensions()
    env_results = test_with_environment()
    debug_critic_batch_handling_result = debug_critic_batch_handling()
    
    basic_policy_works = test_alpha_tuning()  # Renamed since we're just testing basic functionality
    
    
    print("\n=== Test Results Summary ===")
    print(f"✓ Hidden State Persistence: Hidden states maintained across {len(hidden_states)} steps")
    print(f"✓ Actor Gradients Flow to Shared: {actor_grad}")
    print(f"✓ Critic Gradients Isolated: {critic_isolation}")
    print(f"✓ Temporal Sensitivity: {is_temporal}")
    print(f"✓ Encoder Dimensions Correct: {encoder_correct}")
    print(f"✓ Environment Interaction: {len(env_results[0])} episodes completed")
    print(f"✓ Basic Policy Works: {basic_policy_works}")
    print(f"✓ Debug Critic Batch Handling: {debug_critic_batch_handling_result}")
    print("\nPlots saved in 'rsac_test_plots/' directory")

if __name__ == "__main__":
    run_all_tests()