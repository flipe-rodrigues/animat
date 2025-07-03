import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from policy import NumpyStyleRNNPolicy
from dataset import DemonstrationDataset
from train_rnn_simple_noise import train_simple_robust_rnn
from environment import make_arm_env
from shimmy_wrapper import DmControlCompatibilityV0, SelectiveVecNormalize, create_env

def load_trained_rnn(model_path, config_path=None):
    """Load a trained RNN policy."""
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']
    
    # Create policy with same config
    policy = NumpyStyleRNNPolicy(
        obs_dim=config['obs_dim'],
        action_dim=config['action_dim'],
        hidden_size=config['hidden_size'],
        activation=torch.nn.Sigmoid(),
        alpha=1.0/config['tau_mem']
    )
    
    # Load weights
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    
    return policy, config

def collect_rnn_trajectories(env, policy, n_episodes=50, max_steps=50):
    """Roll out the RNN policy to collect states."""
    
    obs_buf = []
    policy.eval()
    device = next(policy.parameters()).device
    
    print(f"Collecting {n_episodes} episodes with RNN...")
    
    for ep in tqdm(range(n_episodes), desc="RNN rollouts"):
        obs = env.reset()
        
        # FIX: Handle vectorized environment - extract first element
        if isinstance(obs, np.ndarray) and obs.ndim > 1:
            obs = obs[0]  # Extract from batch dimension
        
        hidden = policy.init_hidden(1, device=device)
        
        for t in range(max_steps):
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            # Get action from RNN
            with torch.no_grad():
                action, hidden = policy(obs_tensor, hidden)
            
            # Convert action back to numpy
            action = action.cpu().numpy().squeeze(0)
            
            # Store observation (before taking action)
            obs_buf.append(obs.copy())
            
            # Step environment
            next_obs, reward, done, info = env.step(action)
            
            # FIX: Handle vectorized environment output
            if isinstance(next_obs, np.ndarray) and next_obs.ndim > 1:
                obs = next_obs[0]  # Extract from batch dimension
                done = done[0] if hasattr(done, '__len__') else done
            else:
                obs = next_obs
            
            if done:
                break
    
    return np.array(obs_buf)

def expand_demonstrations(original_demos, new_observations, expert_actions):
    """Add new data to existing demonstrations."""
    
    expanded_demos = original_demos.copy()
    
    # Get current data size
    current_size = len(expanded_demos['observations'])
    new_size = len(new_observations)
    
    # Expand observations and actions
    expanded_demos['observations'] = np.vstack([
        expanded_demos['observations'], 
        new_observations
    ])
    expanded_demos['actions'] = np.vstack([
        expanded_demos['actions'], 
        expert_actions
    ])
    
    # Expand other arrays
    expanded_demos['rewards'] = np.concatenate([
        expanded_demos['rewards'],
        np.zeros(new_size)  # Dummy rewards
    ])
    expanded_demos['dones'] = np.concatenate([
        expanded_demos['dones'],
        np.zeros(new_size, dtype=bool)  # Dummy dones
    ])
    
    # Add episode starts (treat new data as individual sequences)
    # For DAgger, we often treat each collected sequence as separate
    new_episode_starts = []
    for i in range(0, new_size, 50):  # Assuming 50-step episodes
        new_episode_starts.append(current_size + i)
    
    expanded_demos['episode_starts'] = np.concatenate([
        expanded_demos['episode_starts'],
        np.array(new_episode_starts)
    ])
    
    return expanded_demos

def dagger_loop(
    initial_rnn_path=None,  # Path to pre-trained RNN
    sac_model_path="arm_final2.zip",
    demonstrations_path="sac_demonstrations_50steps_successful_selective5.pkl",
    selective_norm_path="selective_vec_normalize3.pkl",
    dagger_iters=3,
    episodes_per_iter=50,
    training_epochs_per_iter=50,
    save_dir="./rnn_dagger"
):
    """Main DAgger training loop."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Step 1: Load initial demonstrations
    print("Loading initial demonstrations...")
    with open(demonstrations_path, 'rb') as f:
        demos = pickle.load(f)
    
    print(f"Initial dataset: {len(demos['observations'])} steps")
    
    # Step 2: Get initial RNN policy
    if initial_rnn_path and os.path.exists(initial_rnn_path):
        print(f"Loading pre-trained RNN from {initial_rnn_path}")
        rnn_policy, _ = load_trained_rnn(initial_rnn_path)
    else:
        print("Training initial RNN from scratch...")
        rnn_policy = train_simple_robust_rnn(
            demonstrations_path=demonstrations_path,
            save_dir=os.path.join(save_dir, "initial_rnn"), 
            selective_norm_path=selective_norm_path,
            num_epochs=100,
            noise_std=0.015,
            noise_prob=0.4
        )
        # Save initial RNN
        torch.save(rnn_policy.state_dict(), os.path.join(save_dir, "initial_rnn_weights.pth"))
    
    # Step 3: Load SAC expert
    print(f"Loading SAC expert from {sac_model_path}")
    sac_expert = SAC.load(sac_model_path)
    
    # Step 4: Create environment for data collection
    print("Setting up environment...")
    
    def make_eval_env():
        dm_env = make_arm_env(random_seed=42)
        env = DmControlCompatibilityV0(dm_env)
        from shimmy_wrapper import ModalityEncodingWrapper
        from gymnasium.wrappers import FlattenObservation, RescaleAction
        env = FlattenObservation(env)
        env = ModalityEncodingWrapper(env, grid_size=4)
        env = RescaleAction(env, min_action=0.0, max_action=1.0)
        return env
    
    # Create vectorized environment
    vec_env = DummyVecEnv([make_eval_env])
    
    # Apply normalization
    vec_env = SelectiveVecNormalize.load(selective_norm_path, vec_env)
    vec_env.training = False  # Set to eval mode
    
    # Step 5: DAgger iterations
    current_demos = demos
    
    for iteration in range(dagger_iters):
        print(f"\n{'='*60}")
        print(f"🔄 DAgger Iteration {iteration + 1}/{dagger_iters}")
        print(f"{'='*60}")
        
        # 5a: Collect trajectories with current RNN
        print("Phase 1: Collecting RNN trajectories...")
        
        # Create single environment for collection
        single_env_func = lambda: make_eval_env()
        single_env = single_env_func()
        single_vec = DummyVecEnv([lambda: single_env])
        single_vec = SelectiveVecNormalize.load(selective_norm_path, single_vec)
        single_vec.training = False
        
        # Collect states visited by RNN
        collected_observations = collect_rnn_trajectories(
            single_vec, 
            rnn_policy, 
            n_episodes=episodes_per_iter,
            max_steps=50
        )
        
        single_vec.close()
        
        print(f"Collected {len(collected_observations)} observations")
        
        # 5b: Get expert labels for collected states
        print("Phase 2: Getting expert labels...")
        expert_actions = []
        
        for obs in tqdm(collected_observations, desc="Querying SAC expert"):
            # Query SAC expert (note: SAC expects different normalization)
            # We need to convert from selective norm to original norm
            action, _ = sac_expert.predict(obs.reshape(1, -1), deterministic=True)
            
            # Convert action from [-1,1] to [0,1] to match training data
            normalized_action = (action[0] + 1.0) / 2.0
            normalized_action = np.clip(normalized_action, 0.0, 1.0)
            
            expert_actions.append(normalized_action)
        
        expert_actions = np.array(expert_actions)
        print(f"Generated {len(expert_actions)} expert labels")
        
        # 5c: Expand dataset
        print("Phase 3: Expanding dataset...")
        current_demos = expand_demonstrations(
            current_demos, 
            collected_observations, 
            expert_actions
        )
        
        print(f"Dataset expanded to {len(current_demos['observations'])} steps")
        
        # Save expanded dataset
        expanded_path = os.path.join(save_dir, f"demos_dagger_iter_{iteration + 1}.pkl")
        with open(expanded_path, 'wb') as f:
            pickle.dump(current_demos, f)
        
        # 5d: Retrain RNN on expanded dataset
        print("Phase 4: Retraining RNN...")
        rnn_policy = train_simple_robust_rnn(
            demonstrations_path=expanded_path,
            save_dir=os.path.join(save_dir, f"rnn_iter_{iteration + 1}"),
            selective_norm_path=selective_norm_path,
            num_epochs=training_epochs_per_iter,
            noise_std=0.015,
            noise_prob=0.4
        )
        
        print(f"✅ DAgger iteration {iteration + 1} completed!")
    
    vec_env.close()
    
    print(f"\n🎉 DAgger training completed!")
    print(f"Final dataset size: {len(current_demos['observations'])} steps")
    print(f"Models saved in: {save_dir}")
    
    return rnn_policy

if __name__ == "__main__":
    # Run DAgger starting from your best noise-trained model
    dagger_loop(
        initial_rnn_path="./rnn_simple_noise_test/best_rnn_full.pth",  # Use your trained model
        sac_model_path="arm_final2.zip",
        demonstrations_path="sac_demonstrations_50steps_successful_selective5.pkl",
        selective_norm_path="selective_vec_normalize3.pkl",
        dagger_iters=3,  # Start with 3 iterations
        episodes_per_iter=50,  # Moderate collection size
        training_epochs_per_iter=50,  # Quick retraining
        save_dir="./rnn_dagger_test"
    )