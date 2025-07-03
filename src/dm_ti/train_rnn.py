import os
import torch
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

from tianshou.policy import PPOPolicy
from tianshou.utils import TensorboardLogger
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.trainer import OnpolicyTrainer

from shimmy_wrapper import create_training_env_good, create_eval_env_good, set_seeds
from policy_networks import RecurrentActorNetwork, ModalitySpecificEncoder, CriticNetwork
from torch.distributions import Independent, Normal

# Add this at the module level, outside any function
update_step_counter = 0

def load_checkpoint_if_resume(args, policy, optim, shared_encoder, device):
    """Load checkpoint if resume is enabled"""
    if not args.resume:
        return
    
    # Simple: just try to load from policy_best.pth
    log_path = os.path.join(args.logdir, 'ppo_rnn')
    checkpoint_path = os.path.join(log_path, 'policy_good_final.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ No checkpoint found at {checkpoint_path}")
        print("⚠️  Starting fresh training...")
        return
    
    try:
        print(f"📁 Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Simple loading - assume it's our format
        policy.load_state_dict(checkpoint['policy'], strict=False)
        print("✅ Policy loaded successfully")
        
        # Load optimizer if not resetting
        if 'optim' in checkpoint and not getattr(args, 'reset_optimizer', False):
            optim.load_state_dict(checkpoint['optim'])
            print("✅ Optimizer loaded successfully")
        
        print("🚀 Resume successful!")
        
    except Exception as e:
        print(f"❌ Loading failed: {e}")
        print("⚠️  Starting fresh training...")

def load_checkpoint_if_resume_simple(args, policy, optim, device):
    """Simplified checkpoint loading to avoid parameter mismatch."""
    if not args.resume:
        return
    
    log_path = os.path.join(args.logdir, 'ppo_rnn')
    checkpoint_path = os.path.join(log_path, 'policy_best_fixed.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ No checkpoint found at {checkpoint_path}")
        return
    
    try:
        print(f"📁 Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # **SAFE: Only load policy weights, skip optimizer**
        if 'policy' in checkpoint:
            policy.load_state_dict(checkpoint['policy'], strict=False)
            print("✅ Policy weights loaded successfully")
        elif 'actor' in checkpoint and 'critic' in checkpoint:
            policy.actor.load_state_dict(checkpoint['actor'], strict=False)
            policy.critic.load_state_dict(checkpoint['critic'], strict=False)
            print("✅ Actor and Critic loaded separately")
        else:
            policy.load_state_dict(checkpoint, strict=False)
            print("✅ Policy loaded (legacy format)")
        
        # **SKIP optimizer loading to avoid dimension mismatch**
        print("⚠️  Optimizer reset (avoiding dimension mismatch)")
        print("🚀 Resume successful with fresh optimizer!")
        
    except Exception as e:
        print(f"❌ Loading failed: {e}")
        print("⚠️  Starting fresh training...")

def train_ppo(args):
    # Set random seeds
    set_seeds(args.seed)
    
    # Create log directory
    log_path = os.path.join(args.logdir, 'ppo_rnn')
    os.makedirs(log_path, exist_ok=True)
    
    # Print the exact path for debugging
    absolute_log_path = os.path.abspath(log_path)
    print(f"=== TENSORBOARD LOGS LOCATION ===")
    print(f"Log directory: {absolute_log_path}")
    print(f"TensorBoard command: tensorboard --logdir={absolute_log_path}")
    print("="*50)
    
    # Setup tensorboard writer
    writer = SummaryWriter(log_path)
    
    # Select device
    device = torch.device(args.device)
    
    # Create training and test environments
    train_envs = create_training_env_good(num_envs=args.training_num, base_seed=args.seed)
    test_envs = create_eval_env_good(num_envs=args.test_num, base_seed=args.seed+100)
    
    # Get environment specifications
    obs_shape = train_envs.observation_space[0].shape
    action_shape = train_envs.action_space[0].shape
    
    # Create SHARED encoder (back to the working version)
    shared_encoder = ModalitySpecificEncoder(target_size=40).to(device)
    
    # Create networks with shared encoder
    actor = RecurrentActorNetwork(
        obs_shape=obs_shape,
        action_shape=action_shape,
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

    # **MOVED: Optimizer creation before checkpoint loading**
    shared_params = list(shared_encoder.parameters())
    
    if len(shared_params) == 0:
        print("Using fixed shared encoder (population coding only)")
        all_params = set(actor.parameters()) | set(critic.parameters())
        all_params = list(all_params)
    else:
        print("Using shared encoder with trainable parameters")
        all_params = list(actor.parameters()) + list(critic.parameters())
    
    optim = torch.optim.Adam(all_params, lr=args.actor_lr)
    
    # **MOVED: Create policy before initialization**
    lr_scheduler = None
    if args.lr_decay:
        max_update_num = np.ceil(args.step_per_epoch / args.step_per_collect) * args.epoch
        lr_scheduler = LambdaLR(optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num)
    
    def dist(logits):
        loc, scale = logits
        return Independent(Normal(loc, scale), 1)
    
    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        reward_normalization=args.rew_norm,
        action_scaling=True,
        action_bound_method=args.bound_action_method,
        lr_scheduler=lr_scheduler,
        action_space=train_envs.action_space[0],
        eps_clip=args.eps_clip,
        value_clip=args.value_clip,
        dual_clip=args.dual_clip,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
    )
    
    # **CRITICAL: Load checkpoint FIRST before any initialization**
    if args.resume:
        print("Resume flag detected, loading checkpoint BEFORE initialization...")
        load_checkpoint_if_resume_simple(args, policy, optim, device)
    
    # **ONLY apply initialization if NOT resuming**
    if not args.resume:
        print("Applying Tianshou-style network initialization (fresh training)...")
        
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
    else:
        print("⚠️  Skipping initialization - using loaded weights")
    
    # Print parameter counts for verification
    print(f"Actor total params: {sum(p.numel() for p in actor.parameters()):,}")
    print(f"Critic total params: {sum(p.numel() for p in critic.parameters()):,}")
    print(f"Shared encoder params: {sum(p.numel() for p in shared_encoder.parameters()):,}")
    
    # Simplified monitoring function
    def monitor_training(policy, result=None, batch=None, env_step=None):
        """Unified monitoring function that collects all important statistics."""
        global_step = env_step or update_step_counter
        
        # PPO Algorithm Metrics
        if result:
            metrics = {
                "algorithm/loss_total": getattr(result, "loss", None),
                "algorithm/loss_policy": getattr(result, "policy_loss", None),
                "algorithm/loss_value": getattr(result, "vf_loss", None), 
                "algorithm/entropy": getattr(result, "entropy_loss", None),
                "algorithm/kl_divergence": getattr(result, "approx_kl", None),
                "algorithm/clip_fraction": getattr(result, "clip_fraction", None),
            }
            
            for name, value in metrics.items():
                if value is not None and isinstance(value, (torch.Tensor, float, int)):
                    value = value.item() if isinstance(value, torch.Tensor) else value
                    writer.add_scalar(name, value, global_step=global_step)

    # Replace the update method
    original_update = policy.update

    def update_wrapper(*args, **kwargs):
        global update_step_counter
        
        # Call original update
        result = original_update(*args, **kwargs)
        
        # Increment counter and monitor
        update_step_counter += 1
        monitor_training(policy, result=result, env_step=update_step_counter)
        
        return result

    policy.update = update_wrapper

    # Create buffer with stack_num for RNN experience replay
    buffer = VectorReplayBuffer(
        total_size=args.buffer_size,
        buffer_num=args.training_num,
        stack_num=args.stack_num,
    )
    
    # Create collectors for training and testing
    train_collector = Collector(policy, train_envs, buffer)
    test_collector = Collector(policy, test_envs)
    
    # Create tensorboard logger
    logger = TensorboardLogger(writer)
    
    # **ENHANCED: Save functions with complete state**
    def save_best_fn(policy):
        save_data = {
            'policy': policy.state_dict(),
            'actor': policy.actor.state_dict(),
            'critic': policy.critic.state_dict(),
            'encoder': shared_encoder.state_dict(),
            'optim': optim.state_dict(),  # **NEW: Save optimizer state**
            'gradient_step': update_step_counter,  # **NEW: Save step counter**
            'args': vars(args),  # **NEW: Save training arguments**
        }
        torch.save(save_data, os.path.join(log_path, 'policy_best.pth'))
        return save_data
    
    def save_checkpoint_fn(epoch, env_step, gradient_step):
        if epoch % args.save_interval == 0:
            save_data = {
                'epoch': epoch,
                'env_step': env_step,
                'gradient_step': gradient_step,
                'policy': policy.state_dict(),
                'actor': policy.actor.state_dict(),
                'critic': policy.critic.state_dict(),
                'encoder': shared_encoder.state_dict(),
                'optim': optim.state_dict(),  # **NEW: Save optimizer state**
                'args': vars(args),  # **NEW: Save training arguments**
            }
            torch.save(save_data, os.path.join(log_path, f'checkpoint_{epoch}.pth'))
            return save_data
        return None
    
    # Setup training regime (REMOVED invalid start_epoch parameter)
    result = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        repeat_per_collect=args.repeat_per_collect,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        step_per_collect=args.step_per_collect,
        logger=logger,
        verbose=True,
        test_in_train=False,
        save_best_fn=save_best_fn,
        save_checkpoint_fn=save_checkpoint_fn,
        train_fn=lambda epoch, step: monitor_training(policy, env_step=step)
    ).run()
    
    # Save final policy
    torch.save(
        policy.state_dict(),
        os.path.join(log_path, 'policy.pth')
    )
    
    # Clean up
    train_envs.close()
    test_envs.close()
    
    return result


def make_parser():
    parser = argparse.ArgumentParser()
    # Environment
    parser.add_argument("--training-num", type=int, default=20, help="Number of training envs")  # CHANGED: 20→8
    parser.add_argument("--test-num", type=int, default=10, help="Number of test envs")
    
    # Network
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden size of RNN")
    parser.add_argument("--num-layers", type=int, default=1, help="Number of RNN layers")
    parser.add_argument("--stack-num", type=int, default=8, help="Stack number for RNN in buffer")
    
    # Training
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--buffer-size", type=int, default=4000, help="Buffer size")  # CHANGED: 20000→4096
    parser.add_argument("--actor-lr", type=float, default=3e-4, help="Learning rate")  # Unified LR
    parser.add_argument("--critic-lr", type=float, default=3e-4, help="Critic learning rate (deprecated)")
    parser.add_argument("--lr-decay", action="store_true", help="Enable learning rate decay")  # NEW
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--epoch", type=int, default=100, help="Number of epochs")
    parser.add_argument("--step-per-epoch", type=int, default=20000, help="Steps per epoch")
    parser.add_argument("--step-per-collect", type=int, default=2000, help="Steps per collection")  # CHANGED: 2000→2048
    parser.add_argument("--repeat-per-collect", type=int, default=10, help="Repeat per collection")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max gradient norm")
    
    # PPO specific - ADAPTED from Tianshou example
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")  # CHANGED: 0.5→0.25
    parser.add_argument("--ent-coef", type=float, default=0.0, help="Entropy coefficient")
    parser.add_argument("--eps-clip", type=float, default=0.2, help="Epsilon clip")
    parser.add_argument("--value-clip", type=int, default=1, help="Use value clip")  # CHANGED: 1→0
    parser.add_argument("--dual-clip", type=float, default=None, help="Dual clip")
    parser.add_argument("--norm-adv", type=int, default=1, help="Normalize advantage")
    parser.add_argument("--recompute-adv", type=int, default=1, help="Recompute advantage")
    parser.add_argument("--rew-norm", type=int, default=0, help="Normalize rewards")  # CHANGED: enabled by default
    parser.add_argument("--bound-action-method", type=str, default="clip", help="Action bounding method")  # NEW
    
    # Resume training arguments
    parser.add_argument("--resume", action="store_true", help="Resume training from best checkpoint")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Specific checkpoint path to load")
    
    # Logging
    parser.add_argument("--logdir", type=str, default="logs", help="Log directory path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-interval", type=int, default=4, help="Save interval for checkpoints")
    
    return parser


if __name__ == "__main__":
    # Parse arguments
    args = make_parser().parse_args()
    
    # Print configuration
    print("=== TRAINING CONFIGURATION ===")
    print(f"Resume: {args.resume}")
    if args.checkpoint_path:
        print(f"Checkpoint path: {args.checkpoint_path}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Actor LR: {args.actor_lr}")
    print(f"Epochs: {args.epoch}")
    print("="*30)
    
    # Run training
    result = train_ppo(args)
    
    # Print final result
    print(f"Final reward: {result['best_reward']:.2f}")