import torch
import numpy as np
from tianshou.data import Batch
from shimmy_wrapper import create_env, create_training_env_good, create_eval_env_good, set_seeds
from policy_networks import RecurrentActorNetwork, CriticNetwork, ModalitySpecificEncoder
from tianshou.policy import PPOPolicy
from torch.distributions import Independent, Normal

def create_exact_training_policy(device="cuda" if torch.cuda.is_available() else "cpu"):
    """Create the exact same policy setup as train_rnn.py"""
    # Use same seed as training
    set_seeds(42)
    
    # Create envs exactly like training
    train_envs = create_training_env_good(num_envs=16, base_seed=42)
    
    # Get environment specifications exactly like training
    obs_shape = train_envs.observation_space[0].shape
    action_shape = train_envs.action_space[0].shape
    
    print(f"Environment specs: obs_shape={obs_shape}, action_shape={action_shape}")
    
    # Create shared encoder with EXACT same parameters
    shared_encoder = ModalitySpecificEncoder(target_size=40).to(device)
    
    # Create actor network with EXACT same parameters
    actor = RecurrentActorNetwork(
        obs_shape=obs_shape,
        action_shape=action_shape,
        encoder=shared_encoder,
        hidden_size=64,  # EXACT same as train_rnn.py
        num_layers=1,    # EXACT same
        device=device
    ).to(device)
    
    # Create critic network with EXACT same parameters
    critic = CriticNetwork(
        obs_shape=obs_shape,
        encoder=shared_encoder,
        hidden_size=64,  # EXACT same as train_rnn.py
        device=device
    ).to(device)
    
    # Create optimizer with EXACT same parameters
    optim = torch.optim.Adam([
        {'params': actor.parameters(),  'lr': 1e-4, 'weight_decay': 1e-5},  # EXACT same
        {'params': critic.parameters(), 'lr': 5e-5, 'weight_decay': 1e-5},  # EXACT same
    ])
    
    # Distribution function EXACTLY like training
    def dist(logits):
        mean, sigma = logits
        return Independent(Normal(mean, sigma), 1)
    
    # Create PPO policy with EXACT same parameters
    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist,
        discount_factor=0.99,          # From train_rnn.py
        gae_lambda=0.95,               # From train_rnn.py
        max_grad_norm=1.0,             # From train_rnn.py
        vf_coef=0.5,                   # From train_rnn.py
        ent_coef=0.005,                # From train_rnn.py
        reward_normalization=0,        # From train_rnn.py
        action_scaling=True,
        action_space=train_envs.action_space[0],
        eps_clip=0.2,                  # From train_rnn.py
        value_clip=1,                  # From train_rnn.py
        dual_clip=None,                # From train_rnn.py
        advantage_normalization=1,     # From train_rnn.py
        recompute_advantage=1,         # From train_rnn.py
    )
    
    train_envs.close()
    return policy

def test_shared_encoder_gradients(policy, num_steps=10):
    """Test if shared encoder causes gradient conflicts between actor and critic."""
    print("\n=== Testing Shared Encoder Gradient Flow ===")
    
    # Create dummy batch using EXACT same environment as training
    env = create_env()
    obs = torch.randn(4, env.observation_space.shape[0])
    
    # Test actor gradient flow
    policy.optim.zero_grad()
    (mu, sigma), state = policy.actor(obs)
    actor_loss = mu.sum()
    actor_loss.backward(retain_graph=True)
    
    actor_grads = {}
    for name, param in policy.actor.encoder.named_parameters():
        if param.grad is not None:
            actor_grads[name] = param.grad.clone()
    
    # Test critic gradient flow  
    policy.optim.zero_grad()
    values = policy.critic(obs)
    critic_loss = values.sum()
    critic_loss.backward()
    
    critic_grads = {}
    for name, param in policy.actor.encoder.named_parameters():
        if param.grad is not None:
            critic_grads[name] = param.grad.clone()
    
    # Check for gradient conflicts
    print("Encoder gradient analysis:")
    conflict_detected = False
    for name in actor_grads.keys():
        if name in critic_grads:
            actor_norm = actor_grads[name].norm().item()
            critic_norm = critic_grads[name].norm().item()
            cosine_sim = torch.cosine_similarity(
                actor_grads[name].flatten(), 
                critic_grads[name].flatten(), 
                dim=0
            ).item()
            print(f"  {name}: actor_grad={actor_norm:.6f}, critic_grad={critic_norm:.6f}, cosine_sim={cosine_sim:.4f}")
            
            if cosine_sim < -0.5:
                print(f"    WARNING: Conflicting gradients detected!")
                conflict_detected = True
    
    return conflict_detected

def test_rnn_state_consistency(policy, num_envs=4, seq_len=20):
    """Test RNN state management exactly like training buffer."""
    print("\n=== Testing RNN State Consistency ===")
    
    env = create_env()
    obs_dim = env.observation_space.shape[0]
    
    # Initialize states exactly like training
    initial_state = policy.actor.init_state(num_envs)
    print(f"Initial state shape: {initial_state.shape}")
    
    # Test state evolution with realistic observations
    current_state = initial_state
    obs_seq = torch.randn(seq_len, num_envs, obs_dim)
    
    # Add some realistic observation ranges (from environment bounds)
    obs_seq = torch.clamp(obs_seq * 2.0, -5.0, 5.0)  # Realistic range
    
    nan_detected_step = None
    
    for t in range(seq_len):
        obs_batch = obs_seq[t]
        
        # Check for NaNs in input
        if torch.isnan(obs_batch).any():
            print(f"NaN detected in observation at step {t}")
            
        # Check for NaNs in state
        if torch.isnan(current_state).any():
            print(f"NaN detected in RNN state at step {t}")
            print(f"State stats: min={current_state.min():.6f}, max={current_state.max():.6f}")
            nan_detected_step = t
            break
            
        # Forward pass exactly like training
        try:
            (mu, sigma), new_state = policy.actor(obs_batch, current_state)
            
            # Check outputs for NaNs
            if torch.isnan(mu).any() or torch.isnan(sigma).any():
                print(f"NaN detected in actor output at step {t}")
                print(f"Mu stats: min={mu.min():.6f}, max={mu.max():.6f}")
                print(f"Sigma stats: min={sigma.min():.6f}, max={sigma.max():.6f}")
                nan_detected_step = t
                break
                
            # Check new state
            if torch.isnan(new_state).any():
                print(f"NaN detected in new RNN state at step {t}")
                nan_detected_step = t
                break
                
            current_state = new_state
            
        except Exception as e:
            print(f"Error at step {t}: {e}")
            nan_detected_step = t
            break
    
    return nan_detected_step is None

def test_population_encoder_bounds(policy):
    """Test population encoder with exact input dimensions from training."""
    print("\n=== Testing Population Encoder Bounds ===")
    
    encoder = policy.actor.encoder
    
    # Get exact input dimensions from encoder
    muscle_dim = encoder.muscle_encoder.input_dim if hasattr(encoder, 'muscle_encoder') else 12
    target_dim = encoder.target_encoder.input_dim if hasattr(encoder, 'target_encoder') else 2
    total_dim = muscle_dim + target_dim + 1  # +1 for time
    
    print(f"Encoder expects input dim: {total_dim} (muscle: {muscle_dim}, target: {target_dim})")
    
    # Test with normal inputs
    normal_input = torch.randn(10, total_dim)
    try:
        normal_output = encoder(normal_input)
        print(f"Normal input: output shape={normal_output.shape}, mean={normal_output.mean():.4f}")
        if torch.isnan(normal_output).any():
            print("  NaN detected with normal input!")
            return False
    except Exception as e:
        print(f"Error with normal input: {e}")
        return False
    
    # Test with extreme values that might occur during training
    extreme_tests = [
        ("large_positive", torch.ones_like(normal_input) * 100),
        ("large_negative", torch.ones_like(normal_input) * -100),
        ("mixed_extreme", torch.randn_like(normal_input) * 50),
        ("zeros", torch.zeros_like(normal_input)),
        ("small_values", torch.randn_like(normal_input) * 1e-6),
    ]
    
    nan_detected = False
    for test_name, test_input in extreme_tests:
        try:
            test_output = encoder(test_input)
            if torch.isnan(test_output).any():
                print(f"  NaN detected with {test_name} input!")
                nan_detected = True
        except Exception as e:
            print(f"Error with {test_name} input: {e}")
            nan_detected = True
    
    return not nan_detected

def test_parameter_initialization(policy):
    """Test parameter initialization exactly like training."""
    print("\n=== Testing Parameter Initialization ===")
    
    param_issues = 0
    
    # Check RNN parameters exactly like in networks.py
    if hasattr(policy.actor, 'rnn'):
        rnn = policy.actor.rnn
        print("RNN Parameters:")
        for name, param in rnn.named_parameters():
            if param.data.numel() > 0:
                param_mean = param.data.mean().item()
                param_std = param.data.std().item()
                param_min = param.data.min().item()
                param_max = param.data.max().item()
                
                print(f"  {name}: mean={param_mean:.6f}, std={param_std:.6f}, "
                      f"min={param_min:.6f}, max={param_max:.6f}")
                
                # Check for issues that would cause NaNs
                if abs(param_mean) > 2.0:
                    print(f"    WARNING: Large mean value!")
                    param_issues += 1
                if param_std > 3.0:
                    print(f"    WARNING: Large standard deviation!")
                    param_issues += 1
                if abs(param_min) > 10.0 or abs(param_max) > 10.0:
                    print(f"    WARNING: Extreme parameter values!")
                    param_issues += 1
                    
                # Check for NaN parameters
                if torch.isnan(param).any():
                    print(f"    ERROR: NaN parameters detected!")
                    param_issues += 1
    
    # Check encoder parameters
    print("\nEncoder Parameters:")
    for name, param in policy.actor.encoder.named_parameters():
        if param.data.numel() > 0:
            param_mean = param.data.mean().item()
            param_std = param.data.std().item()
            print(f"  {name}: mean={param_mean:.6f}, std={param_std:.6f}")
            
            if torch.isnan(param).any():
                print(f"    ERROR: NaN parameters in encoder!")
                param_issues += 1
    
    return param_issues == 0

def comprehensive_stability_test(policy, num_tests=100):
    """Test stability with exact training conditions."""
    print("\n=== Comprehensive Stability Test ===")
    
    env = create_env()
    obs_dim = env.observation_space.shape[0]
    batch_size = 16  # Same as training num_envs
    
    nan_detected = False
    error_step = None
    
    for test_idx in range(num_tests):
        # Generate observations similar to training environment
        obs = torch.randn(batch_size, obs_dim) * 2.0  # Realistic scale
        
        # Add some extreme cases that might occur during training
        if test_idx % 10 == 0:
            obs[0, :] = torch.randn(obs_dim) * 10  # Extreme values
        if test_idx % 15 == 0:
            obs[1, :] = torch.zeros(obs_dim)  # Zero values
        if test_idx % 20 == 0:
            obs[2, :] = torch.ones(obs_dim) * 5  # Large positive values
        
        # Test actor exactly like training forward pass
        try:
            state = policy.actor.init_state(batch_size)
            (mu, sigma), new_state = policy.actor(obs, state)
            
            # Test critic too
            values = policy.critic(obs)
            
            # Check for NaNs in all outputs
            if (torch.isnan(mu).any() or torch.isnan(sigma).any() or 
                torch.isnan(new_state).any() or torch.isnan(values).any()):
                print(f"NaN detected in test {test_idx}")
                print(f"  Input stats: mean={obs.mean():.4f}, std={obs.std():.4f}, min={obs.min():.4f}, max={obs.max():.4f}")
                print(f"  Mu has NaN: {torch.isnan(mu).any()}")
                print(f"  Sigma has NaN: {torch.isnan(sigma).any()}")
                print(f"  New state has NaN: {torch.isnan(new_state).any()}")
                print(f"  Values has NaN: {torch.isnan(values).any()}")
                nan_detected = True
                error_step = test_idx
                break
                
        except Exception as e:
            print(f"Exception in test {test_idx}: {e}")
            nan_detected = True
            error_step = test_idx
            break
    
    if not nan_detected:
        print(f"All {num_tests} stability tests passed!")
    
    return not nan_detected

def test_gradient_accumulation(policy, num_steps=10):
    """Test actual gradient accumulation like in training."""
    print("\n=== Testing Gradient Accumulation ===")
    
    env = create_env()
    obs = torch.randn(16, env.observation_space.shape[0])
    
    # Simulate PPO update cycle
    for step in range(num_steps):
        policy.optim.zero_grad()
        
        # Actor loss
        (mu, sigma), state = policy.actor(obs)
        actor_loss = mu.sum()
        actor_loss.backward(retain_graph=True)
        
        # Critic loss  
        values = policy.critic(obs)
        critic_loss = values.sum()
        critic_loss.backward()
        
        # Check gradient norms before clipping
        total_norm = 0
        for param in policy.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        print(f"Step {step}: Total gradient norm = {total_norm:.4f}")
        
        if total_norm > 100:  # Exploding gradients
            print(f"  WARNING: Exploding gradients detected!")
            return False
        
        # Apply gradient clipping like in training
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        
        # Simulate optimizer step
        policy.optim.step()
    
    return True

def test_realistic_sequence_training(policy):
    """Test with realistic sequence lengths from training."""
    print("\n=== Testing Realistic Sequence Training ===")
    
    env = create_env()
    batch_size = 16
    seq_len = 4  # Same as stack_num in training
    obs_dim = env.observation_space.shape[0]
    
    # Create realistic batch with sequence dimension
    obs_batch = torch.randn(batch_size, seq_len, obs_dim)
    
    # Test with different state management scenarios
    scenarios = [
        ("fresh_start", None),
        ("continuing_episode", policy.actor.init_state(batch_size)),
        ("after_done", policy.actor.init_state(batch_size))
    ]
    
    for scenario_name, initial_state in scenarios:
        print(f"Testing scenario: {scenario_name}")
        
        try:
            # Process sequence
            for t in range(seq_len):
                obs_t = obs_batch[:, t]  # [batch, obs_dim]
                (mu, sigma), new_state = policy.actor(obs_t, initial_state)
                values = policy.critic(obs_t)
                
                # Check for NaNs
                if torch.isnan(mu).any() or torch.isnan(values).any():
                    print(f"  NaN detected in {scenario_name} at step {t}")
                    return False
                
                initial_state = new_state
        
        except Exception as e:
            print(f"  Error in {scenario_name}: {e}")
            return False
    
    return True

def test_action_scaling_pipeline(policy):
    """Test the complete action pipeline including scaling."""
    print("\n=== Testing Action Scaling Pipeline ===")
    
    env = create_env()
    obs = torch.randn(16, env.observation_space.shape[0])
    
    try:
        # Get raw action distribution
        (mu, sigma), state = policy.actor(obs)
        
        # Create distribution (like in training)
        from torch.distributions import Independent, Normal
        dist = Independent(Normal(mu, sigma), 1)
        
        # Sample actions (these are in tanh space: [-inf, inf] but mostly [-3, 3])
        raw_actions = dist.sample()
        
        print(f"Raw sampled actions: min={raw_actions.min():.4f}, max={raw_actions.max():.4f}")
        print(f"Raw mu: min={mu.min():.4f}, max={mu.max():.4f}")
        print(f"Raw sigma: min={sigma.min():.4f}, max={sigma.max():.4f}")
        
        # Check for extreme values that could cause issues
        if torch.abs(mu).max() > 5.0:
            print("  WARNING: Large mu values detected!")
            return False
            
        if sigma.max() > 2.0 or sigma.min() < 1e-4:
            print("  WARNING: Extreme sigma values detected!")
            return False
        
        # Test the ACTUAL Tianshou action scaling process
        # This is what happens inside Tianshou's Collector
        if hasattr(env.action_space, 'low') and hasattr(env.action_space, 'high'):
            action_space = env.action_space
            
            # Tianshou's action scaling: map from [-1, 1] to [low, high]
            # But first, raw actions need to be clipped/normalized to [-1, 1]
            
            # Method 1: Tanh squashing (what most continuous control uses)
            tanh_actions = torch.tanh(raw_actions)
            print(f"Tanh actions: min={tanh_actions.min():.4f}, max={tanh_actions.max():.4f}")
            
            # Method 2: Tianshou's actual scaling (from action_space_fn)
            # Scale from [-1, 1] to [low, high]
            action_low = torch.from_numpy(action_space.low).to(raw_actions.device)
            action_high = torch.from_numpy(action_space.high).to(raw_actions.device)
            
            # This is the correct Tianshou scaling formula
            scaled_actions = (tanh_actions + 1.0) / 2.0 * (action_high - action_low) + action_low
            
            print(f"Final scaled actions: min={scaled_actions.min():.4f}, max={scaled_actions.max():.4f}")
            print(f"Action space: low={action_space.low}, high={action_space.high}")
            
            # Now check if scaled actions are in bounds
            if (scaled_actions < action_low).any() or (scaled_actions > action_high).any():
                print("  WARNING: Scaled actions out of bounds!")
                print(f"  Out of bounds by: low={torch.min(scaled_actions - action_low).item():.6f}, high={torch.min(action_high - scaled_actions).item():.6f}")
                return False
            else:
                print("  ✓ Scaled actions are within bounds!")
        
        # Test log probability computation (critical for PPO)
        # Use the original raw_actions, not scaled ones
        try:
            log_prob = dist.log_prob(raw_actions)
            print(f"Log prob: min={log_prob.min():.4f}, max={log_prob.max():.4f}")
            
            if torch.isnan(log_prob).any() or torch.isinf(log_prob).any():
                print("  ERROR: NaN/Inf in log probabilities!")
                return False
                
            if log_prob.min() < -100:  # Very negative log prob
                print("  WARNING: Very negative log probabilities!")
                return False
                
        except Exception as e:
            print(f"  Error computing log probabilities: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error in action scaling: {e}")
        return False

def test_full_training_step(policy):
    """Test a complete training step including all PPO components."""
    print("\n=== Testing Full Training Step ===")
    
    env = create_env()
    batch_size = 16
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    try:
        # Create realistic training batch
        obs = torch.randn(batch_size, obs_dim) * 2.0
        next_obs = torch.randn(batch_size, obs_dim) * 2.0
        actions = torch.randn(batch_size, action_dim) * 0.5  # Tanh-scaled actions
        rewards = torch.randn(batch_size) * 1.0
        dones = torch.zeros(batch_size, dtype=torch.bool)
        dones[::4] = True  # Some episodes done
        
        # Test forward pass
        (mu, sigma), state = policy.actor(obs)
        values = policy.critic(obs)
        next_values = policy.critic(next_obs)
        
        print(f"Forward pass successful:")
        print(f"  Mu range: [{mu.min():.4f}, {mu.max():.4f}]")
        print(f"  Sigma range: [{sigma.min():.4f}, {sigma.max():.4f}]")
        print(f"  Values range: [{values.min():.4f}, {values.max():.4f}]")
        
        # Test distribution creation and sampling
        from torch.distributions import Independent, Normal
        dist = Independent(Normal(mu, sigma), 1)
        
        # Test log probability computation with actual actions
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        
        print(f"  Log prob range: [{log_prob.min():.4f}, {log_prob.max():.4f}]")
        print(f"  Entropy range: [{entropy.min():.4f}, {entropy.max():.4f}]")
        
        # Check for NaNs
        if (torch.isnan(mu).any() or torch.isnan(sigma).any() or 
            torch.isnan(values).any() or torch.isnan(log_prob).any() or
            torch.isnan(entropy).any()):
            print("  ERROR: NaN detected in training step!")
            return False
        
        # Test advantage computation (simplified GAE)
        with torch.no_grad():
            advantages = rewards + 0.99 * next_values.squeeze() * (~dones) - values.squeeze()
            print(f"  Advantages range: [{advantages.min():.4f}, {advantages.max():.4f}]")
            
            if torch.isnan(advantages).any():
                print("  ERROR: NaN in advantages!")
                return False
        
        # Test gradient computation
        policy.optim.zero_grad()
        
        # Simple loss computation
        actor_loss = -log_prob.mean()
        critic_loss = ((values.squeeze() - (rewards + 0.99 * next_values.squeeze() * (~dones))) ** 2).mean()
        entropy_loss = -entropy.mean()
        total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
        
        print(f"  Losses - Actor: {actor_loss:.4f}, Critic: {critic_loss:.4f}, Entropy: {entropy_loss:.4f}")
        
        if torch.isnan(total_loss):
            print("  ERROR: NaN in total loss!")
            return False
        
        # Test backpropagation
        total_loss.backward()
        
        # Check gradients
        grad_norm = 0
        param_count = 0
        nan_grads = 0
        
        for param in policy.parameters():
            if param.grad is not None:
                param_count += 1
                if torch.isnan(param.grad).any():
                    nan_grads += 1
                else:
                    grad_norm += param.grad.data.norm(2).item() ** 2
        
        grad_norm = grad_norm ** 0.5
        
        print(f"  Gradient norm: {grad_norm:.4f}")
        print(f"  Parameters with gradients: {param_count}")
        
        if nan_grads > 0:
            print(f"  ERROR: {nan_grads} parameters have NaN gradients!")
            return False
        
        if grad_norm > 1000:
            print("  WARNING: Very large gradient norm!")
            return False
        
        # Test optimizer step
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        policy.optim.step()
        
        print("  Optimizer step successful")
        
        return True
        
    except Exception as e:
        print(f"Error in full training step: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_complete_network_tests():
    """Run all network stability tests with exact training setup."""
    print("="*60)
    print("COMPLETE NETWORK STABILITY TEST SUITE")
    print("Using EXACT training configuration from train_rnn.py")
    print("="*60)
    
    # Create policy with EXACT same setup as training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = create_exact_training_policy(device)
    
    # Run all tests
    tests_passed = 0
    total_tests = 9  # Updated count
    
    test_results = {}
    
    # Define all tests
    tests = [
        ("parameter_init", test_parameter_initialization, "Parameter initialization test"),
        ("population_encoder", test_population_encoder_bounds, "Population encoder test"),
        ("rnn_consistency", test_rnn_state_consistency, "RNN state consistency test"),
        ("shared_encoder", lambda p: not test_shared_encoder_gradients(p), "Shared encoder gradient test"),
        ("stability", comprehensive_stability_test, "Comprehensive stability test"),
        ("gradient_accumulation", test_gradient_accumulation, "Gradient accumulation test"),
        ("realistic_sequence", test_realistic_sequence_training, "Realistic sequence training test"),
        ("action_scaling", test_action_scaling_pipeline, "Action scaling test"),
        ("full_training_step", test_full_training_step, "Full training step test"),
    ]
    
    # Run each test
    for test_name, test_func, test_description in tests:
        try:
            print(f"\nRunning {test_description}...")
            success = test_func(policy)
            test_results[test_name] = success
            if success:
                tests_passed += 1
                print(f"✓ {test_description}: PASSED")
            else:
                print(f"✗ {test_description}: FAILED")
        except Exception as e:
            print(f"✗ {test_description}: FAILED with exception: {e}")
            test_results[test_name] = False
    
    print("\n" + "="*60)
    print(f"SUMMARY: {tests_passed}/{total_tests} tests passed")
    print("Detailed results:")
    for test_name, passed in test_results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
    print("="*60)
    
    if tests_passed != total_tests:
        print("\nRECOMMENDATIONS:")
        if not test_results.get('parameter_init', True):
            print("- Check network initialization in networks.py")
        if not test_results.get('population_encoder', True):
            print("- Fix population encoder overflow protection")
        if not test_results.get('rnn_consistency', True):
            print("- Fix RNN state handling in policy_networks.py")
        if not test_results.get('shared_encoder', True):
            print("- Consider separate encoders for actor/critic")
        if not test_results.get('stability', True):
            print("- Add gradient clipping and input normalization")
        if not test_results.get('gradient_accumulation', True):
            print("- Investigate gradient accumulation logic")
        if not test_results.get('realistic_sequence', True):
            print("- Ensure sequence handling matches training conditions")
        if not test_results.get('action_scaling', True):
            print("- Verify action scaling implementation")
        if not test_results.get('full_training_step', True):
            print("- Fix issues in complete training pipeline")
    else:
        print("\n🎉 All tests passed! Your network should be stable for training.")
        print("\nIf training still fails, the issue is likely in hyperparameters:")
        print("- Try learning rate: 3e-4 for both actor and critic")
        print("- Try larger batch size: 256 instead of 128")
        print("- Try larger buffer: 8192 instead of 4096")
        print("- Try longer sequences: stack_num=8 instead of 4")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = run_complete_network_tests()
    if success:
        print("\n🚀 Ready for training!")
    else:
        print("\n🔧 Please fix the failed tests before training.")