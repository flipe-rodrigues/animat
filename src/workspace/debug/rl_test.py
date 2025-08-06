"""Test script for RL training with different encoder combinations."""

import torch
import numpy as np
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
import time

from wrappers.rl_wrapper import create_training_env, create_eval_env, set_seeds
from encoders.encoders import IdentityEncoder, ModalitySpecificEncoder, GridEncoder

def test_rl_environment_creation():
    """Test RL environment creation with different encoders."""
    print("🔧 TESTING RL ENVIRONMENT CREATION")
    print("=" * 60)
    
    results = {}
    
    # Test environments with different encoders
    encoder_configs = [
        ("identity", IdentityEncoder(obs_dim=15)),
        ("grid_3x3", ModalitySpecificEncoder(grid_size=3, raw_obs_dim=15)),
        ("grid_5x5", ModalitySpecificEncoder(grid_size=5, raw_obs_dim=15)),
        ("grid_7x7", ModalitySpecificEncoder(grid_size=7, raw_obs_dim=15)),
    ]
    
    for name, encoder in encoder_configs:
        try:
            print(f"\n📊 Testing {name} encoder...")
            
            # Create training environment
            train_env = create_training_env(num_envs=2, encoder=encoder)
            print(f"   ✅ Training env: {train_env.observation_space}")
            
            # Create eval environment
            eval_env = create_eval_env(training_env=train_env, encoder=encoder)
            print(f"   ✅ Eval env: {eval_env.observation_space}")
            
            # Test environment step
            obs = train_env.reset()
            action = train_env.action_space.sample()
            obs_next, reward, done, info = train_env.step(action)
            
            print(f"   ✅ Environment step successful")
            print(f"      Obs shape: {obs.shape}")
            print(f"      Action shape: {action.shape}")
            print(f"      Reward: {reward}")
            
            results[name] = {
                'obs_dim': obs.shape[-1],
                'action_dim': action.shape[-1],
                'encoder_output': encoder.output_dim,
                'status': 'success'
            }
            
            # Clean up
            train_env.close()
            eval_env.close()
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results[name] = {'status': 'failed', 'error': str(e)}
    
    return results

def test_rl_training_short():
    """Test short RL training runs with different algorithms and encoders."""
    print("\n🏃 TESTING SHORT RL TRAINING")
    print("=" * 60)
    
    set_seeds(42)
    
    # Test configurations
    test_configs = [
        ("SAC_identity", SAC, IdentityEncoder(obs_dim=15)),
        ("SAC_grid", SAC, ModalitySpecificEncoder(grid_size=3, raw_obs_dim=15)),
        ("PPO_identity", PPO, IdentityEncoder(obs_dim=15)),
        ("PPO_grid", PPO, ModalitySpecificEncoder(grid_size=3, raw_obs_dim=15)),
    ]
    
    results = {}
    
    for config_name, algorithm, encoder in test_configs:
        try:
            print(f"\n🔥 Testing {config_name}...")
            
            # Create environments
            train_env = create_training_env(num_envs=2, encoder=encoder)
            eval_env = create_eval_env(training_env=train_env, encoder=encoder)
            
            # Create model
            if algorithm == SAC:
                model = SAC(
                    "MlpPolicy", 
                    train_env,
                    verbose=0,
                    learning_rate=3e-4,
                    buffer_size=1000,  # Small for testing
                    learning_starts=100,
                    batch_size=32,
                    tau=0.005,
                    gamma=0.99,
                )
            else:  # PPO
                model = PPO(
                    "MlpPolicy",
                    train_env,
                    verbose=0,
                    learning_rate=3e-4,
                    n_steps=64,  # Small for testing
                    batch_size=32,
                    n_epochs=3,
                    gamma=0.99,
                )
            
            print(f"   📈 Model created, starting training...")
            
            # Short training
            start_time = time.time()
            model.learn(total_timesteps=500)  # Very short for testing
            training_time = time.time() - start_time
            
            # Quick evaluation
            obs = eval_env.reset()
            total_reward = 0
            for _ in range(10):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                total_reward += reward[0]
                if done.any():
                    break
            
            print(f"   ✅ Training completed in {training_time:.2f}s")
            print(f"      Average reward: {total_reward/10:.4f}")
            
            results[config_name] = {
                'training_time': training_time,
                'avg_reward': total_reward/10,
                'obs_dim': encoder.output_dim,
                'status': 'success'
            }
            
            # Clean up
            train_env.close()
            eval_env.close()
            del model
            
        except Exception as e:
            print(f"   ❌ Training failed: {e}")
            results[config_name] = {'status': 'failed', 'error': str(e)}
    
    return results

if __name__ == "__main__":
    print("🚀 RL TRAINING COMPREHENSIVE TEST")
    print("=" * 80)
    
    # Test environment creation
    env_results = test_rl_environment_creation()
    
    # Test short training
    training_results = test_rl_training_short()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 RESULTS SUMMARY")
    print("=" * 80)
    
    print("\n🔧 Environment Creation:")
    for name, result in env_results.items():
        status = "✅" if result['status'] == 'success' else "❌"
        if result['status'] == 'success':
            print(f"  {status} {name}: {result['obs_dim']}D obs, {result['action_dim']}D action")
        else:
            print(f"  {status} {name}: {result['error']}")
    
    print("\n🏃 Training Tests:")
    for name, result in training_results.items():
        status = "✅" if result['status'] == 'success' else "❌"
        if result['status'] == 'success':
            print(f"  {status} {name}: {result['training_time']:.2f}s, reward={result['avg_reward']:.4f}")
        else:
            print(f"  {status} {name}: {result['error']}")
    
    # Overall assessment
    total_tests = len(env_results) + len(training_results)
    passed_tests = sum(1 for r in env_results.values() if r['status'] == 'success') + \
                   sum(1 for r in training_results.values() if r['status'] == 'success')
    
    print(f"\n🎉 OVERALL: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("✅ All RL tests passed! Your system is ready for full training.")
    else:
        print("❌ Some tests failed. Check the errors above.")