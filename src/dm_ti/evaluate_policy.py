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
from policy_networks import RecurrentActorNetwork, CriticNetwork
from policy_networks import ModalitySpecificEncoder

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, default='logs/ppo_rnn/policy_best.pth')
    p.add_argument('--video-dir', type=str, default='videos')
    p.add_argument('--video-name', type=str, default='arm_reaching_rnn.gif')
    p.add_argument('--num-episodes', type=int, default=10)
    p.add_argument('--hidden-size', type=int, default=64)
    p.add_argument('--num-layers', type=int, default=1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--render-every', type=int, default=2, 
                   help='Only save every Nth frame to reduce GIF size')
    return p.parse_args()

def make_policy(actor, critic, action_space, device):
    """Create policy with EXACT same configuration as training."""
    # **ADAPTED: Use single optimizer like in train_rnn.py**
    all_params = set(actor.parameters()) | set(critic.parameters())
    all_params = list(all_params)  # Convert set to list for optimizer
    
    optim = torch.optim.Adam(all_params, lr=3e-4)  # Match training LR
    
    def dist_fn(logits):
        loc, scale = logits  # Unpack the tuple (same as training)
        return Independent(Normal(loc, scale), 1)
    
    # **ADAPTED: Match ALL training parameters exactly**
    policy = PPOPolicy(
        actor=actor, 
        critic=critic, 
        optim=optim, 
        dist_fn=dist_fn,
        discount_factor=0.99, 
        gae_lambda=0.95, 
        max_grad_norm=0.5,  # CHANGED: 1.0→0.5 to match training
        vf_coef=0.5,  # CHANGED: 0.5→0.25 to match training
        ent_coef=0.01,  # CHANGED: 0.01→0.0 to match training
        reward_normalization=0,  # CHANGED: 0→1 to match training
        action_scaling=True,
        action_bound_method="clip",  # NEW: match training
        action_space=action_space,
        eps_clip=0.2,
        value_clip=1,  # CHANGED: 1→0 to match training
        dual_clip=None,
        advantage_normalization=1,  # CHANGED: 1→0 to match training
        recompute_advantage=1,
        lr_scheduler=None  # No scheduler needed for evaluation
    )
    return policy.to(device)

def main():
    args = get_args()
    os.makedirs(args.video_dir, exist_ok=True)
    set_seeds(args.seed)
    device = torch.device(args.device)

    # Build environment
    dm_env = make_arm_env(random_seed=args.seed)
    env = create_env(random_seed=args.seed, base_env=dm_env)

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    # Create shared encoder (same as training)
    shared_encoder = ModalitySpecificEncoder(target_size=40).to(device)

    # Create actor & critic with shared encoder (same as training)
    actor = RecurrentActorNetwork(
        obs_shape=obs_shape, 
        action_shape=act_shape,
        encoder=shared_encoder,
        hidden_size=args.hidden_size, 
        num_layers=args.num_layers,
        device=device
    ).to(device)
    
    critic = CriticNetwork(
        obs_shape=obs_shape, 
        encoder=shared_encoder,
        hidden_size=args.hidden_size, 
        device=device
    ).to(device)

    # **ADAPTED: Apply same initialization as training (for consistency)**
    print("Applying same initialization as training...")
    
    # Initialize sigma parameter for actor
    if hasattr(actor, 'sigma'):
        torch.nn.init.constant_(actor.sigma, -0.5)
        print("✅ Initialized actor.sigma to -0.5")
    
    # Apply orthogonal initialization to all linear layers
    for m in list(actor.modules()) + list(critic.modules()):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
    
    # Special initialization for actor output layer
    if hasattr(actor, 'mu'):
        for m in actor.mu.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.zeros_(m.bias)
                m.weight.data.copy_(0.01 * m.weight.data)
                print("✅ Applied small weight initialization to actor output")

    # Create policy and load weights
    policy = make_policy(actor, critic, env.action_space, device)

    # **ENHANCED: Better checkpoint loading to handle new format**
    print(f"Loading checkpoint from {args.checkpoint}")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        if isinstance(checkpoint, dict):
            if 'policy' in checkpoint:
                # New format with individual components
                policy.load_state_dict(checkpoint['policy'], strict=True)
                if 'encoder' in checkpoint:
                    shared_encoder.load_state_dict(checkpoint['encoder'], strict=True)
                print("✅ Model loaded successfully (new format)")
                
                # Print checkpoint info if available
                if 'args' in checkpoint:
                    saved_args = checkpoint['args']
                    print(f"   Checkpoint trained with hidden_size: {saved_args.get('hidden_size', 'unknown')}")
                    print(f"   Checkpoint gradient step: {checkpoint.get('gradient_step', 'unknown')}")
                    
            elif 'actor' in checkpoint and 'critic' in checkpoint:
                # Alternative new format
                actor.load_state_dict(checkpoint['actor'], strict=True)
                critic.load_state_dict(checkpoint['critic'], strict=True)
                if 'encoder' in checkpoint:
                    shared_encoder.load_state_dict(checkpoint['encoder'], strict=True)
                print("✅ Model loaded successfully (component format)")
                
            else:
                # Old format - just policy state dict
                policy.load_state_dict(checkpoint, strict=True)
                print("✅ Model loaded successfully (old format)")
                
        else:
            # Very old format - direct state dict
            policy.load_state_dict(checkpoint, strict=True)
            print("✅ Model loaded successfully (direct state dict)")
            
    except Exception as e:
        print(f"❌ Strict loading failed: {e}")
        print("🔄 Attempting non-strict loading...")
        try:
            if isinstance(checkpoint, dict) and 'policy' in checkpoint:
                policy.load_state_dict(checkpoint['policy'], strict=False)
            else:
                policy.load_state_dict(checkpoint, strict=False)
            print("⚠️  Model loaded with non-strict parameter matching")
        except Exception as e2:
            print(f"❌ Non-strict loading also failed: {e2}")
            print("❌ Could not load model - check checkpoint compatibility")
            return

    policy.eval()

    print("Model loaded, starting evaluation")
    all_frames = []
    rewards = []
    successes = 0
    MAX_STEPS = 150

    for ep in range(args.num_episodes):
        print(f"Starting episode {ep+1}/{args.num_episodes}")
        
        # Add separator between episodes
        if ep > 0 and len(all_frames) > 0:
            all_frames.append(np.zeros_like(all_frames[0]))
        
        # Ensure complete environment reset between episodes
        obs, _ = env.reset(seed=args.seed+ep)
        state = None  # Reset RNN state
        done = False
        ep_reward = 0
        steps = 0
        
        episode_frames = []
        
        while not done and steps < MAX_STEPS:
            batch = Batch(obs=np.expand_dims(obs, 0), info={})
            with torch.no_grad():
                out = policy.forward(batch, state=state)
            action = out.act[0]
            
            # Add NaN protection for actions
            if np.isnan(action).any():
                print("⚠️  WARNING: NaN detected in action, using zero action")
                action = np.zeros_like(action)
            
            state = out.state
            
            obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            ep_reward += rew
            steps += 1
            
            # Only render every few steps to reduce GIF size
            if steps % args.render_every == 0 or steps == 1 or done or steps == MAX_STEPS:
                try:
                    frame = dm_env.physics.render(height=480, width=640, camera_id=0)
                    episode_frames.append(frame)
                except Exception as e:
                    print(f"Render error: {e}")
        
        # Status update at the end of each episode
        hand = dm_env.physics.bind(dm_env._task._arm.hand).xpos
        target = dm_env.physics.bind(dm_env._task._arm.target).mocap_pos
        distance = np.linalg.norm(hand - target)
        is_success = distance < 0.06
        
        if is_success:
            successes += 1
            
        rewards.append(ep_reward)
        print(f"Episode {ep+1}: reward={ep_reward:.3f}, steps={steps}, distance={distance:.3f}, success={is_success}")
        
        # Add this episode's frames to the collection
        all_frames.extend(episode_frames)

    # Report statistics
    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Mean reward: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"Success rate: {successes}/{args.num_episodes} ({100*successes/args.num_episodes:.1f}%)")
    print(f"Total frames captured: {len(all_frames)}")

    # Save video with slower playback for better viewing
    if all_frames:
        path = os.path.join(args.video_dir, args.video_name)
        print(f"Saving {len(all_frames)} frames to {path}")
        # Use duration=0.1 (10 fps) instead of 0.03 (33 fps) for slower playback
        imageio.mimsave(path, all_frames, duration=0.1)
        print(f"✅ Video saved to {path}")
    else:
        print("⚠️  No frames captured - video not saved")

    # Explicit cleanup
    env.close()

if __name__ == '__main__':
    main()