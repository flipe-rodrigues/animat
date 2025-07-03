import os
# Add this at the very top of your file, before any other imports
os.environ['MUJOCO_GL'] = 'osmesa'  # Use software rendering instead of hardware

import torch
import numpy as np
import argparse
import imageio
from tianshou.data import Batch

# Import environment creation functions directly
from environment import make_arm_env
from shimmy_wrapper import create_env, set_seeds
# Import SAC policy instead of RNN policy
from bio_policy_sac2 import create_simple_sac, sync_hidden_states


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate trained SAC bio-policy and record video')
    parser.add_argument('--checkpoint', type=str, default='models_rsac/policy.pth', 
                        help='Path to the SAC checkpoint file')
    parser.add_argument('--video-dir', type=str, default='videos_sac', 
                        help='Directory to save videos')
    parser.add_argument('--video-name', type=str, default='arm_reaching_sac.gif',
                        help='Output video filename')
    parser.add_argument('--num-episodes', type=int, default=3, 
                        help='Number of episodes to evaluate')
    parser.add_argument('--hidden-size', type=int, default=64, 
                        help='Hidden layer size (must match training)')
    parser.add_argument('--tau-mem', type=float, default=8.0, 
                        help='Membrane time constant (must match training)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cpu', 
                        help='Device to run on')
    parser.add_argument('--render-height', type=int, default=480,
                        help='Video height')
    parser.add_argument('--render-width', type=int, default=640,
                        help='Video width')
    return parser.parse_args()


def main():
    # Get arguments
    args = get_args()
    
    # Create directories for videos
    os.makedirs(args.video_dir, exist_ok=True)
    
    # Set random seeds for reproducibility
    set_seeds(args.seed)
    
    print("Creating DM Control environment directly...")
    # Create the DM Control environment FIRST
    dm_env = make_arm_env(random_seed=args.seed)
    
    # Then create the wrapped environment using the same dm_env instance
    env = create_env(random_seed=args.seed, base_env=dm_env)
    
    # Now we have direct access to dm_env for rendering and state tracking
    print("Successfully created environment with rendering access")
    
    # Get environment shapes
    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    action_space = env.action_space
    
    print(f"Creating SAC policy with observation dim {state_shape[0]} and action dim {action_shape[0]}")
    
    # Create policy with the same architecture as during training
    policy = create_simple_sac(
        obs_dim=state_shape[0],
        action_dim=action_shape[0],
        action_space=action_space,
        hidden_size=args.hidden_size,
        tau_mem=args.tau_mem,
        device=args.device,
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            policy.load_state_dict(checkpoint['model'])
            epoch = checkpoint.get('epoch', 'unknown')
            print(f"Loaded checkpoint from epoch {epoch}")
        else:
            # Assume the checkpoint is just the model state dict
            policy.load_state_dict(checkpoint)
            print("Loaded model weights directly (no epoch info)")
    except FileNotFoundError:
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Running with randomly initialized model")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Running with randomly initialized model")
    
    # Switch policy to evaluation mode
    policy.eval()
    
    # Prepare for video recording
    all_frames = []
    
    # Run evaluation
    print(f"Evaluating for {args.num_episodes} episodes...")
    rewards = []
    success_count = 0
    
    for episode in range(args.num_episodes):
        print(f"Episode {episode + 1}/{args.num_episodes}")
        
        # Reset environment and get initial observation
        obs, _ = env.reset(seed=args.seed + episode)
        terminated = False
        truncated = False
        state = None
        episode_reward = 0
        timesteps = 0
        
        # Reset policy hidden states at the beginning of each episode
        sync_hidden_states(policy)
        
        # Collect frames for this episode
        episode_frames = []
        
        # Add separator frames between episodes
        if episode > 0:
            for _ in range(10):  # 10 black frames as separator
                black_frame = np.zeros((args.render_height, args.render_width, 3), dtype=np.uint8)
                all_frames.append(black_frame)
        
        while not (terminated or truncated):
            # Format observation for policy
            batch_obs = np.expand_dims(obs, axis=0)
            batch = Batch(
                obs=batch_obs,
                info={}
            )
            
            # Get action from policy (force deterministic for evaluation)
            with torch.no_grad():
                result = policy(batch, state=state, deterministic=True)
                action = result.act[0].cpu().numpy()  # Get first action from batch
                state = result.state if hasattr(result, 'state') else None
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            timesteps += 1
            
            # Render frame directly from dm_env's physics
            frame = dm_env.physics.render(
                height=args.render_height,
                width=args.render_width,
                camera_id=0  # Use the camera defined in XML
            )
            episode_frames.append(frame)
            
            # Print stats and track hand/target every 20 steps
            if timesteps % 20 == 0:
                print(f"  Step {timesteps}, reward: {reward:.4f}")
                
                # Access hand position and velocity directly
                hand_pos = dm_env.physics.bind(dm_env._task._arm.hand).xpos
                hand_vel = dm_env.physics.bind(dm_env._task._arm.hand).cvel[3:6]  # linear velocity
                target_pos = dm_env.physics.bind(dm_env._task._arm.target).mocap_pos
                
                distance = np.linalg.norm(hand_pos - target_pos)
                speed = np.linalg.norm(hand_vel)
                
                print(f"  Hand position: {hand_pos}")
                print(f"  Hand velocity: {hand_vel}, Speed: {speed:.4f}")
                print(f"  Target position: {target_pos}")
                print(f"  Distance to target: {distance:.4f}")
        
        rewards.append(episode_reward)
        
        # Track successful episodes (adjust threshold as needed)
        hand_pos = dm_env.physics.bind(dm_env._task._arm.hand).xpos
        target_pos = dm_env.physics.bind(dm_env._task._arm.target).mocap_pos
        final_distance = np.linalg.norm(hand_pos - target_pos)
        success = final_distance < 0.1  # Using 0.1 as threshold for "reaching"
        if success:
            success_count += 1
        
        print(f"  Episode finished: Reward={episode_reward:.4f}, Steps={timesteps}, " 
              f"Final Distance={final_distance:.4f}, Success={success}")
        
        # Add this episode's frames to the collection
        all_frames.extend(episode_frames)
    
    # Report statistics
    print("\nEvaluation Results:")
    print(f"Number of episodes: {args.num_episodes}")
    print(f"Mean reward: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")
    print(f"Success rate: {success_count}/{args.num_episodes} ({100*success_count/args.num_episodes:.1f}%)")
    
    # Save video
    video_path = os.path.join(args.video_dir, args.video_name)
    imageio.mimsave(video_path, all_frames, duration=33)  # 33ms ≈ 30fps (1000/30)
    print(f"Video saved to: {video_path}")
    
    # Close environment
    env.close()


if __name__ == "__main__":
    main()