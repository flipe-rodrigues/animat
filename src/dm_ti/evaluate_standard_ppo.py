import os
# Add this at the very top of your file, before any other imports
os.environ['MUJOCO_GL'] = 'osmesa'  # Use software rendering instead of hardware

import torch
import numpy as np
import argparse
import imageio
from tianshou.data import Batch
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic

# Import environment and encoding components
from environment import make_arm_env
from shimmy_wrapper import create_env, set_seeds
from bio_networks import ModalitySpecificEncoder
from train_standard_ppo import NormalizedEncodingWrapper  # Use NormalizedEncodingWrapper instead of EncodingWrapper


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate a trained standard PPO policy and record video')
    parser.add_argument('--checkpoint', type=str, default='models/encoded_ppo/encoded_ppo_best.pth', 
                        help='Path to the checkpoint file')
    parser.add_argument('--video-dir', type=str, default='videos', 
                        help='Directory to save videos')
    parser.add_argument('--video-name', type=str, default='standard_ppo_reaching.gif',
                        help='Output video filename')
    parser.add_argument('--num-episodes', type=int, default=5, 
                        help='Number of episodes to evaluate')
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=[64, 64],
                        help='Hidden layer sizes (must match training)')
    parser.add_argument('--seed', type=int, default=40, 
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cpu', 
                        help='Device to run on')
    parser.add_argument('--render-height', type=int, default=480,
                        help='Video height')
    parser.add_argument('--render-width', type=int, default=640,
                        help='Video width')
    return parser.parse_args()


def create_standard_policy(state_shape, action_shape, max_action, hidden_sizes, device, action_space):
    """Create a standard MLP-based PPO policy with encoding."""
    # Create encoder - EXACTLY as in train_standard_ppo.py
    encoder = ModalitySpecificEncoder(
        target_size=39,  # FIXED: match the encoder used in training
        device=device
    )
    
    encoded_state_shape = encoder.output_size
    input_norm = torch.nn.LayerNorm(encoded_state_shape)  # FIXED: Add normalizer as in training
    
    # Actor network
    net_a = Net(
        encoded_state_shape, 
        hidden_sizes=hidden_sizes,
        activation=torch.nn.Tanh,
        device=device
    )
    actor_base = ActorProb(
        net_a, 
        action_shape,
        max_action=max_action,
        unbounded=True,
        device=device
    )
    # Wrap with normalized encoder to match training
    actor = NormalizedEncodingWrapper(actor_base, encoder, input_norm)
   
    # Critic network
    net_c = Net(
        encoded_state_shape, 
        hidden_sizes=hidden_sizes,
        activation=torch.nn.Tanh,
        device=device
    )
    critic_base = Critic(net_c, device=device)
    # Wrap with normalized encoder to match training
    critic = NormalizedEncodingWrapper(critic_base, encoder, input_norm)
    
    # Create optimizer (not needed for evaluation but PPOPolicy requires it)
    optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), 
        lr=3e-4
    )
    
    # Create PPO policy with EXACTLY the same parameters as training
    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        # FIXED: Use the same dist_fn as in training
        dist_fn=lambda x: torch.distributions.Independent(
            torch.distributions.Normal(*x), 1
        ),
        discount_factor=0.99,
        eps_clip=0.2,
        vf_coef=0.5,
        ent_coef=0.01,  # FIXED: match training value (0.01 not 0.02)
        action_space=action_space,  # Pass the actual action space
        action_scaling=True,
        reward_normalization=False,
        advantage_normalization=True,
        recompute_advantage=True,
        value_clip=True,
    )
    
    return policy


def main():
    # Get arguments
    args = get_args()
    
    # Create directories for videos
    os.makedirs(args.video_dir, exist_ok=True)
    
    # Set random seeds for reproducibility
    set_seeds(args.seed)
    
    print("Creating DM Control environment directly...")
    # IMPORTANT: Create the DM Control environment FIRST
    dm_env = make_arm_env(random_seed=args.seed)
    
    # Then create the wrapped environment using the same dm_env instance
    env = create_env(random_seed=args.seed, base_env=dm_env)
    
    # Now we have direct access to dm_env for rendering and state tracking
    print("Successfully created environment with rendering access")
    
    # Get environment shapes
    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    max_action = env.action_space.high[0]
    action_space = env.action_space
    
    print(f"Creating standard PPO policy with observation dim {state_shape[0]} and action dim {action_shape[0]}")
    
    # Create policy with the same architecture as during training
    policy = create_standard_policy(
        state_shape=state_shape[0],
        action_shape=action_shape[0],
        max_action=max_action,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
        action_space=action_space  # Pass the action_space
    )
    
    # Load checkpoint with better error handling
    print(f"Loading checkpoint from {args.checkpoint}")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                policy.load_state_dict(checkpoint['model'])
                print("Loaded checkpoint with 'model' key")
            elif 'state_dict' in checkpoint:
                policy.load_state_dict(checkpoint['state_dict'])
                print("Loaded checkpoint with 'state_dict' key")
            else:
                # Try direct load as a last resort
                policy.load_state_dict(checkpoint)
                print("Loaded checkpoint dictionary directly")
        else:
            # Assume the checkpoint is just the model state dict
            policy.load_state_dict(checkpoint)
            print("Loaded model weights directly")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("WARNING: Running with randomly initialized model!")
    
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
        episode_reward = 0
        timesteps = 0
        
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
            
            # Get action from policy
            with torch.no_grad():  # Ensure no gradients are computed during evaluation
                result = policy.forward(batch)
                action = result.act[0]
            
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
                target_pos = dm_env.physics.bind(dm_env._task._arm.target).mocap_pos
                
                distance = np.linalg.norm(hand_pos - target_pos)
                
                print(f"  Hand position: {hand_pos}")
                print(f"  Target position: {target_pos}")
                print(f"  Distance to target: {distance:.4f}")
        
        rewards.append(episode_reward)
        
        # Get final hand and target positions
        hand_pos = dm_env.physics.bind(dm_env._task._arm.hand).xpos
        target_pos = dm_env.physics.bind(dm_env._task._arm.target).mocap_pos
        final_distance = np.linalg.norm(hand_pos - target_pos)
        
        # Track successful episodes (adjust threshold as needed)
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