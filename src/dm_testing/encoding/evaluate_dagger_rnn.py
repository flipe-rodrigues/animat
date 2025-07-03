import os
os.environ['MUJOCO_GL'] = 'osmesa'

import torch
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from environment import make_arm_env
from shimmy_wrapper import DmControlCompatibilityV0, SelectiveVecNormalize, ModalityEncodingWrapper, set_seeds
from gymnasium.wrappers import FlattenObservation, RescaleAction
from policy import NumpyStyleRNNPolicy
import torch.nn as nn

def load_dagger_rnn(model_path):
    """Load DAgger-trained RNN model."""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    
    # Create model with saved config
    policy = NumpyStyleRNNPolicy(
        obs_dim=config['obs_dim'],
        action_dim=config['action_dim'],
        hidden_size=config['hidden_size'],
        activation=nn.Sigmoid(),
        alpha=1.0/config['tau_mem']
    )
    
    # Load weights
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    
    print(f"✅ Loaded DAgger RNN model:")
    print(f"   Training loss: {checkpoint.get('loss', 'N/A'):.6f}")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Hidden size: {config['hidden_size']}")
    print(f"   Noise training: {config.get('noise_training', False)}")
    print(f"   Noise std: {config.get('noise_std', 'N/A')}")
    print(f"   Training data: {config.get('training_data', 'N/A')}")
    
    return policy, config

def create_evaluation_env(random_seed=42):
    """Create environment exactly like DAgger training."""
    dm_env = make_arm_env(random_seed=random_seed)
    env = DmControlCompatibilityV0(dm_env)
    env = FlattenObservation(env)
    env = ModalityEncodingWrapper(env, grid_size=4)
    env = RescaleAction(env, min_action=0.0, max_action=1.0)
    return env

def evaluate_dagger_model(
    model_path="./rnn_dagger_test/rnn_iter_3/best_rnn_simple_noise.pth",
    norm_path="selective_vec_normalize3.pkl",
    num_episodes=30,
    max_steps=50
):
    """Evaluate DAgger model with proper environment setup."""
    
    print(f"🎯 Evaluating DAgger RNN Model")
    print(f"=" * 40)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy, config = load_dagger_rnn(model_path)
    policy = policy.to(device)
    
    # Create environment exactly like training
    def make_eval_env():
        return create_evaluation_env(random_seed=42)
    
    vec_env = DummyVecEnv([make_eval_env])
    vec_env = SelectiveVecNormalize.load(norm_path, vec_env)
    vec_env.training = False
    
    print(f"✅ Environment setup complete")
    
    # Evaluation loop
    successful_episodes = 0
    total_episodes = 0
    episode_rewards = []
    final_distances = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        set_seeds(42 + episode)
        
        obs = vec_env.reset()
        hidden = policy.init_hidden(1, device=device)
        done = False
        episode_reward = 0.0
        step_count = 0
        
        # Get initial distance
        dm_env = vec_env.envs[0].env.env.env  # Navigate to dm_control env
        hand_pos = dm_env.physics.bind(dm_env._task._arm.hand).xpos
        target_pos = dm_env.physics.bind(dm_env._task._arm.target).mocap_pos
        initial_distance = np.linalg.norm(hand_pos - target_pos)
        
        while not done and step_count < max_steps:
            # RNN forward pass
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action_tensor, hidden = policy(obs_tensor, hidden)
                action = action_tensor.cpu().numpy()
            
            # Step environment
            obs, reward, done, info = vec_env.step(action)
            episode_reward += reward[0]
            step_count += 1
            
            done = done[0] if isinstance(done, np.ndarray) else done
        
        # Final distance
        hand_pos = dm_env.physics.bind(dm_env._task._arm.hand).xpos
        target_pos = dm_env.physics.bind(dm_env._task._arm.target).mocap_pos
        final_distance = np.linalg.norm(hand_pos - target_pos)
        
        # Success check
        is_success = final_distance < 0.06
        if is_success:
            successful_episodes += 1
        
        total_episodes += 1
        episode_rewards.append(episode_reward)
        final_distances.append(final_distance)
        episode_lengths.append(step_count)
        
        print(f"Episode {episode+1:2d}: "
              f"Steps={step_count:2d}, "
              f"Reward={episode_reward:6.2f}, "
              f"Final_dist={final_distance:.4f}, "
              f"Success={'✅' if is_success else '❌'}")
    
    vec_env.close()
    
    # Results
    success_rate = successful_episodes / total_episodes
    avg_reward = np.mean(episode_rewards)
    avg_distance = np.mean(final_distances)
    avg_length = np.mean(episode_lengths)
    
    print(f"\n📊 DAGGER EVALUATION RESULTS:")
    print(f"=" * 40)
    print(f"Success rate: {successful_episodes}/{total_episodes} ({success_rate:.1%})")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average final distance: {avg_distance:.4f}")
    print(f"Average episode length: {avg_length:.1f}")
    
    if success_rate < 0.5:
        print(f"\n⚠️  LOW SUCCESS RATE DETECTED!")
        print(f"Possible issues:")
        print(f"  - Action space mismatch")
        print(f"  - Environment setup difference")
        print(f"  - Normalization issues")
        print(f"  - DAgger training problems")
    
    return success_rate, avg_reward, avg_distance

if __name__ == "__main__":
    success_rate, avg_reward, avg_distance = evaluate_dagger_model(
        #model_path="./rnn_dagger_test/rnn_iter_3/best_rnn_simple_noise.pth",
        model_path="./rnn_simple_noise_test/best_rnn_full.pth",
        norm_path="selective_vec_normalize3.pkl",
        num_episodes=30,
        max_steps=50
    )