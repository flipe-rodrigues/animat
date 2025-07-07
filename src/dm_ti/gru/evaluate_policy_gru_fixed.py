import os
# make sure software rendering works
os.environ['MUJOCO_GL'] = 'osmesa'

import argparse
import imageio
import numpy as np
import torch
from torch.distributions import Independent, Normal
from tianshou.data import Batch
from tianshou.policy import PPOPolicy

from environment import make_arm_env
from shimmy_wrapper import create_env, set_seeds
from policy_networks_gru_fixed import GRUActorNetwork
from policy_networks import CriticNetwork, ModalitySpecificEncoder

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, default='logs/ppo_gru_fixed/policy_best.pth')
    p.add_argument('--video-dir', type=str, default='videos')
    p.add_argument('--video-name', type=str, default='arm_reaching_gru_fixed.gif')
    p.add_argument('--num-episodes', type=int, default=5)
    p.add_argument('--hidden-size', type=int, default=64)
    p.add_argument('--num-layers', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--render-every', type=int, default=3, 
                   help='Only save every Nth frame to reduce GIF size')
    return p.parse_args()

def make_policy(actor, critic, action_space, device):
    """Create policy with EXACT same configuration as training."""
    all_params = set(actor.parameters()) | set(critic.parameters())
    optim = torch.optim.AdamW(
        all_params,
        lr=1e-4,  # MATCH train_rnn_gru_fixed.py
        weight_decay=1e-4
    )
    
    def dist_fn(logits):
        mu, sigma = logits
        return Independent(Normal(mu, sigma), 1)
    
    policy = PPOPolicy(
        actor=actor, 
        critic=critic, 
        optim=optim, 
        dist_fn=dist_fn,
        discount_factor=0.99,
        gae_lambda=0.95,
        max_grad_norm=1.0,  # MATCH train_rnn_gru_fixed.py
        vf_coef=0.5,
        ent_coef=0.02,  # MATCH train_rnn_gru_fixed.py
        reward_normalization=0,
        action_scaling=True,
        action_space=action_space,
        eps_clip=0.2,
        value_clip=1,
        dual_clip=None,
        advantage_normalization=1,
        recompute_advantage=1,
    )
    return policy.to(device)

def main():
    args = get_args()
    os.makedirs(args.video_dir, exist_ok=True)
    set_seeds(args.seed)
    device = torch.device(args.device)

    # FIXED: Build environment the SAME way as evaluate_policy.py
    dm_env = make_arm_env(random_seed=args.seed)
    env = create_env(random_seed=args.seed, base_env=dm_env)

    # FIXED: Get shapes the SAME way as evaluate_policy.py
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    print(f"Environment created - obs_shape: {obs_shape}, act_shape: {act_shape}")

    # Create shared encoder (same as training)
    shared_encoder = ModalitySpecificEncoder(target_size=40).to(device)

    # Create GRU actor
    actor = GRUActorNetwork(
        obs_shape=obs_shape,
        action_shape=act_shape,
        encoder=shared_encoder,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        device=device
    ).to(device)
    
    # Create critic
    critic = CriticNetwork(
        obs_shape=obs_shape,
        encoder=shared_encoder,
        device=device
    ).to(device)

    # Create policy with EXACT same configuration as training
    policy = make_policy(actor, critic, env.action_space, device)

    # Load checkpoint - SAME approach as evaluate_policy.py
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        if isinstance(checkpoint, dict) and 'policy' in checkpoint:
            # New format with individual components
            policy.load_state_dict(checkpoint['policy'], strict=True)
            if 'encoder' in checkpoint:
                shared_encoder.load_state_dict(checkpoint['encoder'])
            print("GRU model loaded successfully (new format)")
        else:
            # Old format - just policy state dict
            policy.load_state_dict(checkpoint, strict=True)
            print("GRU model loaded successfully (old format)")
            
    except Exception as e:
        print(f"Strict loading failed: {e}")
        print("Attempting non-strict loading...")
        if isinstance(checkpoint, dict) and 'policy' in checkpoint:
            policy.load_state_dict(checkpoint['policy'], strict=False)
        else:
            policy.load_state_dict(checkpoint, strict=False)
        print("GRU model loaded with non-strict parameter matching")

    policy.eval()

    print("GRU model loaded, starting evaluation")
    all_frames = []
    rewards = []
    successes = 0
    MAX_STEPS = 150

    for ep in range(args.num_episodes):
        print(f"Starting episode {ep+1}/{args.num_episodes}")
        
        # Add separator between episodes
        if ep > 0 and len(all_frames) > 0:
            all_frames.append(np.zeros_like(all_frames[0]))
        
        # FIXED: Reset environment the SAME way as evaluate_policy.py
        obs, _ = env.reset(seed=args.seed+ep)
        state = None  # Reset GRU state for new episode
        done = False
        ep_reward = 0
        steps = 0
        
        episode_frames = []
        
        while not done and steps < MAX_STEPS:
            # Prepare batch for policy (SAME as evaluate_policy.py)
            batch = Batch(obs=np.expand_dims(obs, 0), info={})
            
            with torch.no_grad():
                out = policy.forward(batch, state=state)
            
            action = out.act[0]  # Extract action from batch
            
            # NaN protection (SAME as evaluate_policy.py)
            if np.isnan(action).any():
                print("WARNING: NaN detected in action, using zero action")
                action = np.zeros_like(action)
                
            state = out.state  # Update GRU state
            
            # FIXED: Step environment the SAME way as evaluate_policy.py
            obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            ep_reward += rew
            steps += 1
            
            # FIXED: Render the SAME way as evaluate_policy.py
            if steps % args.render_every == 0 or steps == 1 or done or steps == MAX_STEPS:
                try:
                    frame = dm_env.physics.render(height=480, width=640, camera_id=0)
                    episode_frames.append(frame)
                except Exception as e:
                    print(f"Render error: {e}")
        
        # FIXED: Calculate final metrics the SAME way as evaluate_policy.py
        hand = dm_env.physics.bind(dm_env._task._arm.hand).xpos
        target = dm_env.physics.bind(dm_env._task._arm.target).mocap_pos
        distance = np.linalg.norm(hand - target)
        is_success = distance < 0.1
        
        if is_success:
            successes += 1
            
        rewards.append(ep_reward)
        print(f"Episode {ep+1}: reward={ep_reward:.3f}, steps={steps}, distance={distance:.3f}, success={is_success}")
        
        # Add this episode's frames to the collection
        all_frames.extend(episode_frames)

    # Report statistics
    print(f"\nGRU Results:")
    print(f"Mean reward: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"Success rate: {successes}/{args.num_episodes} ({100*successes/args.num_episodes:.1f}%)")

    # Save video (SAME as evaluate_policy.py)
    if len(all_frames) > 0:
        path = os.path.join(args.video_dir, args.video_name)
        print(f"Saving {len(all_frames)} frames to {path}")
        imageio.mimsave(path, all_frames, duration=0.1)
        print(f"GRU video saved to {path}")
    else:
        print("Warning: No frames to save")

    # Cleanup (SAME as evaluate_policy.py)
    env.close()

if __name__ == '__main__':
    main()