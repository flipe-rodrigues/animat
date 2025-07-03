import os
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer, Batch
from tianshou.policy import BasePolicy
from shimmy_wrapper import create_training_env_good, create_eval_env_good, create_env, set_seeds

class SimpleRandomPolicy(BasePolicy):
    """Simple random policy that works with any action space."""
    
    def __init__(self, action_space, observation_space=None):
        super().__init__(action_space=action_space, observation_space=observation_space)
        self.action_space = action_space
    
    def forward(self, batch, state=None, **kwargs):
        """Generate random actions for the given batch."""
        batch_size = len(batch.obs)
        actions = []
        
        for _ in range(batch_size):
            action = self.action_space.sample()
            actions.append(action)
        
        return Batch(act=np.array(actions), state=state)
    
    def learn(self, batch, **kwargs):
        """Random policy doesn't learn."""
        return {}

def analyze_observation_structure(obs, verbose=False):
    """Analyze the structure of observations to verify encoding."""
    if verbose:
        print(f"Total observation shape: {obs.shape}")
    
    # Based on ModalitySpecificEncoder structure:
    # 12 sensory (z-scored) + 80 encoded target (2*40) + 1 z-coordinate = 93
    sensory_data = obs[:12]  # First 12 elements
    encoded_target = obs[12:92]  # Next 80 elements (2*40 population encoding)
    z_coord = obs[92]  # Last element (z-coordinate)
    
    analysis = {
        'sensory_stats': {
            'mean': np.mean(sensory_data),
            'std': np.std(sensory_data),
            'min': np.min(sensory_data),
            'max': np.max(sensory_data),
            'range': np.max(sensory_data) - np.min(sensory_data)
        },
        'encoded_target_stats': {
            'mean': np.mean(encoded_target),
            'std': np.std(encoded_target),
            'min': np.min(encoded_target),
            'max': np.max(encoded_target),
            'nonzero_count': np.count_nonzero(encoded_target),
            'max_activation': np.max(encoded_target)
        },
        'z_coord': z_coord,
        'total_obs_shape': obs.shape[0]
    }
    
    if verbose:
        print(f"Sensory data (12 elements): mean={analysis['sensory_stats']['mean']:.3f}, "
              f"std={analysis['sensory_stats']['std']:.3f}, "
              f"range=[{analysis['sensory_stats']['min']:.3f}, {analysis['sensory_stats']['max']:.3f}]")
        print(f"Encoded target (80 elements): mean={analysis['encoded_target_stats']['mean']:.3f}, "
              f"max_activation={analysis['encoded_target_stats']['max_activation']:.3f}, "
              f"nonzero={analysis['encoded_target_stats']['nonzero_count']}")
        print(f"Z coordinate: {analysis['z_coord']:.6f}")
    
    return analysis

def check_observation_constraints(obs_analysis):
    """Check if observations meet the expected constraints."""
    issues = []
    
    # Check sensory data normalization (should be roughly z-scored)
    sensory_stats = obs_analysis['sensory_stats']
    if abs(sensory_stats['mean']) > 0.5:  # Should be roughly zero-mean
        issues.append(f"Sensory mean too high: {sensory_stats['mean']:.3f} (expected ~0)")
    
    if sensory_stats['std'] < 0.5 or sensory_stats['std'] > 2.0:  # Should be roughly unit variance
        issues.append(f"Sensory std unusual: {sensory_stats['std']:.3f} (expected ~1)")
    
    # Check Z coordinate (should be 0 or close to 0)
    if abs(obs_analysis['z_coord']) > 0.1:
        issues.append(f"Z coordinate not zero: {obs_analysis['z_coord']:.6f} (expected ~0)")
    
    # Check encoded target activations (should be between 0 and 1 for population encoding)
    target_stats = obs_analysis['encoded_target_stats']
    if target_stats['min'] < -0.1 or target_stats['max'] > 1.1:
        issues.append(f"Encoded target out of range [0,1]: [{target_stats['min']:.3f}, {target_stats['max']:.3f}]")
    
    # Check observation shape
    if obs_analysis['total_obs_shape'] != 93:
        issues.append(f"Wrong observation shape: {obs_analysis['total_obs_shape']} (expected 93)")
    
    return issues

def test_single_environment_observation_structure():
    """Test observation structure in a single environment."""
    print("="*60)
    print("TESTING SINGLE ENVIRONMENT OBSERVATION STRUCTURE")
    print("="*60)
    
    env = create_env(random_seed=42)
    
    print(f"Environment observation space: {env.observation_space}")
    print(f"Environment action space: {env.action_space}")
    
    # Test multiple resets and steps to check consistency
    for episode in range(3):
        print(f"\n--- Episode {episode} ---")
        obs, info = env.reset()
        
        analysis = analyze_observation_structure(obs, verbose=True)
        issues = check_observation_constraints(analysis)
        
        if issues:
            print("❌ ISSUES FOUND:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✅ All observation constraints satisfied")
        
        # Take a few steps and check consistency
        for step in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            step_analysis = analyze_observation_structure(obs)
            step_issues = check_observation_constraints(step_analysis)
            
            print(f"  Step {step}: reward={reward:.4f}, "
                  f"sensory_mean={step_analysis['sensory_stats']['mean']:.3f}, "
                  f"z_coord={step_analysis['z_coord']:.6f}, "
                  f"issues={len(step_issues)}")
    
    env.close()

def test_shimmy_wrapper():
    """Test if shimmy wrapper and collectors work properly with correct observation structure."""
    print("\n" + "="*60)
    print("DEBUGGING SHIMMY WRAPPER & OBSERVATION STRUCTURE")
    print("="*60)
    
    # Set base seed
    set_seeds(42)
    
    # Test 1: Check observation structure across multiple environments
    print("\n1. Testing Observation Structure Across Environments...")
    
    num_envs = 4
    train_envs = create_training_env_good(num_envs=num_envs, base_seed=12345)
    
    # Reset all environments and check observations
    obs, info = train_envs.reset()
    print(f"Number of training environments: {len(obs)}")
    print(f"Observation shape per env: {obs[0].shape}")
    
    # Analyze observations from all environments
    all_issues = []
    for i, ob in enumerate(obs):
        analysis = analyze_observation_structure(ob)
        issues = check_observation_constraints(analysis)
        
        print(f"Env {i}: sensory_mean={analysis['sensory_stats']['mean']:.3f}, "
              f"sensory_std={analysis['sensory_stats']['std']:.3f}, "
              f"z_coord={analysis['z_coord']:.6f}, "
              f"obs_shape={analysis['total_obs_shape']}")
        
        if issues:
            all_issues.extend([f"Env {i}: {issue}" for issue in issues])
    
    if all_issues:
        print("❌ OBSERVATION ISSUES FOUND:")
        for issue in all_issues:
            print(f"  - {issue}")
    else:
        print("✅ All environments have correct observation structure")
    
    # Test 2: Check if observations change properly during episode
    print(f"\n2. Testing Observation Changes During Episodes...")
    
    initial_analyses = []
    for i in range(num_envs):
        initial_analyses.append(analyze_observation_structure(obs[i]))
    
    # Take 5 steps and analyze changes
    for step in range(5):
        actions = []
        for i in range(num_envs):
            action = train_envs.action_space[i].sample()
            actions.append(action)
        
        obs, rewards, terminated, truncated, infos = train_envs.step(np.array(actions))
        
        # Check if sensory data is changing (muscle sensors should change)
        sensory_changes = []
        for i in range(num_envs):
            current_analysis = analyze_observation_structure(obs[i])
            initial_sensory = obs[i][:12]  # Current sensory
            
            # Compare with some reasonable change threshold
            sensory_change = np.mean(np.abs(initial_sensory))
            sensory_changes.append(sensory_change)
        
        print(f"Step {step}: rewards={[f'{r:.3f}' for r in rewards]}, "
              f"avg_sensory_activity={np.mean(sensory_changes):.3f}")
    
    # Test 3: Verify target encoding consistency
    print(f"\n3. Testing Target Encoding Consistency...")
    
    # Reset and check if target encoding is consistent
    obs, info = train_envs.reset()
    
    # Extract encoded target portions
    encoded_targets = []
    z_coords = []
    for i in range(num_envs):
        encoded_target = obs[i][12:92]  # 80-dimensional encoded target
        z_coord = obs[i][92]
        encoded_targets.append(encoded_target)
        z_coords.append(z_coord)
    
    # Check if all environments have the same target (they should if using fixed target)
    target_similarities = []
    for i in range(1, num_envs):
        similarity = np.corrcoef(encoded_targets[0], encoded_targets[i])[0, 1]
        target_similarities.append(similarity)
    
    print(f"Target encoding similarities: {[f'{s:.3f}' for s in target_similarities]}")
    print(f"Z coordinates: {[f'{z:.6f}' for z in z_coords]}")
    
    if all(abs(z) < 0.1 for z in z_coords):
        print("✅ All Z coordinates are near zero")
    else:
        print("❌ Some Z coordinates are not near zero")
    
    if all(s > 0.95 for s in target_similarities):  # High correlation expected for same target
        print("✅ Target encodings are consistent across environments")
    else:
        print("⚠️  Target encodings vary across environments (this might be expected)")
    
    # Test 4: Test collectors with observation validation
    print(f"\n4. Testing Collectors with Observation Validation...")

    policy = SimpleRandomPolicy(
        action_space=train_envs.action_space[0],
        observation_space=train_envs.observation_space[0]
    )

    buffer = VectorReplayBuffer(total_size=1000, buffer_num=num_envs)
    collector = Collector(policy, train_envs, buffer)

    print("Collecting 100 steps...")
    try:
        collector.reset()
        result = collector.collect(n_step=100)

        # Handle different possible return types
        if hasattr(result, 'n_ep'):
            print(f"Collection successful:")
            print(f"  Episodes: {result.n_ep}")
            print(f"  Steps: {result.n_st}")
            print(f"  Mean reward: {result.rew}")
        else:
            print(f"Collection completed (result type: {type(result)})")

        # Validate observations in buffer
        if len(buffer) > 0:
            try:
                sample_size = min(32, len(buffer))
                batch = buffer.sample(sample_size)
                
                # Handle case where batch might be a tuple
                if isinstance(batch, tuple):
                    print(f"Buffer returned tuple, using first element")
                    batch = batch[0] if batch else None
                
                if batch is not None and hasattr(batch, 'obs'):
                    print(f"\nBuffer observation validation:")
                    print(f"  Buffer obs shape: {batch.obs.shape}")
                    
                    # Check a few observations from buffer
                    buffer_issues = []
                    for i in range(min(5, sample_size)):
                        obs_sample = batch.obs[i]
                        analysis = analyze_observation_structure(obs_sample)
                        issues = check_observation_constraints(analysis)
                        
                        if issues:
                            buffer_issues.extend([f"Sample {i}: {issue}" for issue in issues])
                    
                    if buffer_issues:
                        print("❌ Buffer observation issues:")
                        for issue in buffer_issues:
                            print(f"    - {issue}")
                    else:
                        print("✅ Buffer observations are correctly structured")
                else:
                    print("⚠️ Unable to validate buffer observations - unexpected batch format")
                    
            except Exception as buffer_error:
                print(f"⚠️ Buffer validation failed: {buffer_error}")

    except Exception as e:
        print(f"❌ Collection failed: {e}")
        import traceback
        traceback.print_exc()

    # Cleanup
    train_envs.close()
    
    print(f"\n" + "="*60)
    print("DEBUGGING COMPLETE")
    print("="*60)

def test_normalization_files():
    """Test if normalization files exist and have reasonable values."""
    print("\n" + "="*60)
    print("TESTING NORMALIZATION FILES")
    print("="*60)
    
    import pickle
    
    # Check if stats files exist
    stats_dir = '/home/afons/animat/mujoco'
    sensor_stats_path = os.path.join(stats_dir, 'sensor_stats.pkl')
    target_stats_path = os.path.join(stats_dir, 'hand_position_stats.pkl')
    
    if os.path.exists(sensor_stats_path):
        with open(sensor_stats_path, 'rb') as f:
            sensor_stats = pickle.load(f)
        print(f"✅ Sensor stats loaded:")
        print(f"  Mean: {sensor_stats['mean']}")
        print(f"  Std: {sensor_stats['std']}")
        print(f"  Shape: mean={np.array(sensor_stats['mean']).shape}, std={np.array(sensor_stats['std']).shape}")
    else:
        print(f"❌ Sensor stats file not found: {sensor_stats_path}")
    
    if os.path.exists(target_stats_path):
        with open(target_stats_path, 'rb') as f:
            target_stats = pickle.load(f)
        print(f"✅ Target stats loaded:")
        print(f"  Mean: {target_stats['mean']}")
        print(f"  Std: {target_stats['std']}")
        print(f"  Shape: mean={np.array(target_stats['mean']).shape}, std={np.array(target_stats['std']).shape}")
    else:
        print(f"❌ Target stats file not found: {target_stats_path}")

if __name__ == "__main__":
   
    # Run all tests
    test_single_environment_observation_structure()
    test_shimmy_wrapper()
    test_normalization_files()
    
    print("\nAll tests completed!")