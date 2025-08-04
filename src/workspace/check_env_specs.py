"""Check actual environment specifications."""
import numpy as np
from envs.dm_env import make_arm_env

def check_env_specs():
    """Check the actual environment specifications."""
    env = make_arm_env(random_seed=42)
    
    # Check action space
    action_spec = env.action_spec()
    print(f"Action space: {action_spec}")
    print(f"Action shape: {action_spec.shape}")
    print(f"Action minimum: {action_spec.minimum}")
    print(f"Action maximum: {action_spec.maximum}")
    
    # Check observation space
    timestep = env.reset()
    print(f"\nObservation keys: {list(timestep.observation.keys())}")
    
    for key, value in timestep.observation.items():
        print(f"  {key}: shape={value.shape}, type={type(value)}")
    
    # Test with correct action size
    correct_action = np.zeros(action_spec.shape)
    print(f"\nTesting with correct action shape: {correct_action.shape}")
    
    try:
        timestep = env.step(correct_action)
        print(f"✅ Step successful with correct action size!")
        print(f"Reward: {timestep.reward}")
    except Exception as e:
        print(f"❌ Step failed: {e}")
    
    env.close()
    return action_spec.shape[0]

if __name__ == "__main__":
    action_dim = check_env_specs()
    print(f"\n🎯 Use action_dim = {action_dim} in your ES trainer!")