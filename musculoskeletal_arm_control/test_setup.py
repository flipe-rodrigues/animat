"""
Quick test script to verify environment setup.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from envs.reaching_env import ReachingEnv
from utils.place_cells import PlaceCellGrid


def test_place_cells():
    """Test place cell encoding."""
    print("Testing place cells...")
    
    workspace_bounds = ((0.1, 0.6), (0.3, 0.7))
    grid = PlaceCellGrid(
        workspace_bounds=workspace_bounds,
        grid_size=(8, 8),
        sigma=0.08
    )
    
    # Test encoding
    position = np.array([0.35, 0.5])
    activities = grid.encode(position)
    
    print(f"  Grid size: {grid.grid_size}")
    print(f"  Number of cells: {grid.num_cells}")
    print(f"  Test position: {position}")
    print(f"  Activities shape: {activities.shape}")
    print(f"  Max activity: {activities.max():.3f}")
    print(f"  Sum of activities: {activities.sum():.3f}")
    
    # Test decoding
    decoded_pos = grid.decode(activities)
    print(f"  Decoded position: {decoded_pos}")
    print(f"  Decoding error: {np.linalg.norm(decoded_pos - position):.6f}")
    
    print("✓ Place cells working correctly\n")


def test_environment():
    """Test environment initialization and basic functionality."""
    print("Testing environment...")
    
    env = ReachingEnv(
        render_mode=None,
        hold_time_range=(0.2, 0.8),
        reach_threshold=0.03,
        hold_threshold=0.04,
        max_episode_steps=1000
    )
    
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")
    
    # Test reset
    obs, info = env.reset()
    print(f"  Initial observation shape: {obs.shape}")
    print(f"  Target position: {info['target_position']}")
    print(f"  Required hold time: {info['required_hold_time']:.2f}s")
    
    # Test step with random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"  After step:")
    print(f"    Observation shape: {obs.shape}")
    print(f"    Reward: {reward:.2f}")
    print(f"    Distance to target: {info['distance_to_target']:.3f}m")
    
    # Test multiple steps
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    
    env.close()
    print("✓ Environment working correctly\n")


def test_policies():
    """Test policy initialization."""
    print("Testing policies...")
    
    from agents.mlp_policy import MLPPolicy
    from agents.rnn_policy import RNNPolicy
    
    obs_dim = 77
    action_dim = 4
    
    # Test MLP
    mlp = MLPPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=(256, 256),
        activation="relu"
    )
    
    test_obs = np.random.randn(obs_dim)
    action = mlp.predict(test_obs)
    print(f"  MLP output shape: {action.shape}")
    print(f"  MLP output range: [{action.min():.3f}, {action.max():.3f}]")
    
    # Test RNN
    rnn = RNNPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_size=128,
        num_layers=1
    )
    
    action, hidden = rnn.predict(test_obs)
    print(f"  RNN output shape: {action.shape}")
    print(f"  RNN output range: [{action.min():.3f}, {action.max():.3f}]")
    print(f"  RNN hidden state shape: {hidden[0].shape}")
    
    print("✓ Policies working correctly\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Running Environment Tests")
    print("=" * 60)
    print()
    
    try:
        test_place_cells()
        test_environment()
        test_policies()
        
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        print("\nYou can now run:")
        print("  python train.py       # Train the models")
        print("  python visualize.py   # Visualize trained policies")
        
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
