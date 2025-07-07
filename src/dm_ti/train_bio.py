import os
import torch
import torch.nn as nn
import numpy as np
import tianshou as ts
from tianshou.policy import PPOPolicy
from tianshou.utils import TensorboardLogger
from tianshou.trainer import OnpolicyTrainer
from tianshou.data import Collector, VectorReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import argparse

from shimmy_wrapper import create_training_env, create_eval_env, create_env, set_seeds
from bio_policy_rnn import make_rnn_actor_critic


def get_args():
    parser = argparse.ArgumentParser()
    # Training parameters
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--train-num', type=int, default=16, help='number of training environments')
    parser.add_argument('--test-num', type=int, default=10, help='number of test environments')
    parser.add_argument('--buffer-size', type=int, default=20000, help='replay buffer size')
    parser.add_argument('--step-per-epoch', type=int, default=5000, help='steps per epoch')
    parser.add_argument('--step-per-collect', type=int, default=1024, help='steps per collection')
    parser.add_argument('--repeat-per-collect', type=int, default=6, help='repeat training per collection')
    
    # Network parameters
    parser.add_argument('--hidden-size', type=int, default=128, help='hidden layer size')
    parser.add_argument('--tau-mem', type=float, default=16.0, help='membrane time constant')
    parser.add_argument('--tau-adapt', type=float, default=70.0, help='adaptation time constant')
    
    # Logging and saving
    parser.add_argument('--log-dir', type=str, default='logs', help='log directory')
    parser.add_argument('--save-dir', type=str, default='models', help='model save directory')
    parser.add_argument('--save-interval', type=int, default=10, help='model save interval')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Learning rate scheduler parameters
    parser.add_argument('--lr-decay', type=float, default=0.5, help='learning rate decay factor')
    parser.add_argument('--lr-decay-epochs', type=int, default=150, help='epochs before decay')
    parser.add_argument('--use-scheduler', action='store_true', help='use learning rate scheduler')
    parser.add_argument('--use-onecycle', action='store_true', 
                        help='use OneCycle learning rate scheduler')
    
    return parser.parse_args()


def main():
    # Get arguments
    args = get_args()
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set random seeds
    set_seeds(args.seed)
    
    # Create environments
    train_envs = create_training_env(num_envs=args.train_num)
    test_envs = create_eval_env(num_envs=args.test_num) 
    
    # Create temporary evaluation env to get observation and action spaces
    temp_env = create_env()
    
    # Get environment shapes
    state_shape = temp_env.observation_space.shape
    action_shape = temp_env.action_space.shape
    action_space = temp_env.action_space
    
    # Define scheduler factory with OneCycleLR
    scheduler_factory = None
    if args.use_scheduler:
        if args.use_onecycle:
            def scheduler_factory(optim):
                total_steps = args.epochs * args.step_per_epoch
                return torch.optim.lr_scheduler.OneCycleLR(
                    optim,
                    max_lr=args.lr,
                    total_steps=total_steps,
                    pct_start=0.1,          # 10% warmup (gentler)
                    div_factor=10,          # Start at lr/10 (higher)
                    final_div_factor=100,   # End at lr/100 (not as extreme)
                    anneal_strategy='cos'
                )
        else:
            def scheduler_factory(optim):
                return torch.optim.lr_scheduler.StepLR(
                    optim, 
                    step_size=args.lr_decay_epochs, 
                    gamma=args.lr_decay
                )
    
    # Create policy with optional scheduler
    policy = make_rnn_actor_critic(
        obs_dim=state_shape[0],
        action_dim=action_shape[0],
        action_space=action_space,
        hidden_size=args.hidden_size,
        tau_mem=args.tau_mem,
        tau_adapt=args.tau_adapt,
        adapt_scale=0.05,  # Use args.adapt_scale if available
        lr=args.lr,
        device=args.device,
        scheduler_factory=scheduler_factory
    )
    
    # Create buffer
    buffer = VectorReplayBuffer(
        args.buffer_size,
        len(train_envs)
    )
    
    # Use Collector without state_info_keys
    train_collector = Collector(
        policy,
        train_envs,
        buffer
    )
    test_collector = Collector(
        policy,
        test_envs
    )
    
    # Create logger
    writer = SummaryWriter(args.log_dir)
    logger = TensorboardLogger(writer)
    
    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(args.save_dir, 'policy.pth'))
    
    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # Save checkpoint every save_interval epochs
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'model': policy.state_dict(),
                'optim': policy.optim.state_dict(),
                'epoch': epoch,
                'env_step': env_step,
                'gradient_step': gradient_step,
            }, os.path.join(args.save_dir, f"checkpoint_{epoch}.pth"))
    
    def find_latest_checkpoint(save_dir):
        """Find the latest checkpoint in the save directory."""
        checkpoints = [f for f in os.listdir(save_dir) if f.startswith("checkpoint_") and f.endswith(".pth")]
        if not checkpoints:
            return None
        checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        return os.path.join(save_dir, checkpoints[-1])
    
    def train_fn(epoch, env_step):
        # Reset RNN state at the beginning of each epoch
        if env_step % args.step_per_epoch == 0:
            print(f"Resetting hidden states for epoch {epoch+1}")
            policy.actor.reset_state(batch_size=args.train_num)
        
        """Monitor for critical stability issues and apply emergency interventions only when necessary"""
        # Initialize storage for tracking metrics
        if not hasattr(train_fn, "loss_history"):
            train_fn.loss_history = []
            train_fn.high_loss_count = 0
        
        # ===== 1. EMERGENCY LOSS EXPLOSION DETECTION =====
        # Extract latest loss from logger if available
        if hasattr(logger, "writer") and hasattr(logger.writer, "_writer"):
            for tag in ["train/loss", "train/loss/clip"]:
                scalar_dict = logger.writer._writer.scalars
                if tag in scalar_dict and len(scalar_dict[tag]) > 0:
                    loss_val = scalar_dict[tag][-1][2]  # Get last value
                    if isinstance(loss_val, (int, float)) and not np.isnan(loss_val):
                        train_fn.loss_history.append(loss_val)
                        train_fn.loss_history = train_fn.loss_history[-20:]  # Keep last 20
                        
                        # ONLY intervene on catastrophic loss explosions
                        if loss_val > 10000:
                            train_fn.high_loss_count += 1
                            print(f"🚨 CRITICAL LOSS EXPLOSION: {loss_val:.2f}")
                            
                            # Emergency intervention only after multiple high losses
                            if train_fn.high_loss_count >= 3:
                                print("🔄 EMERGENCY RNN STATE RESET")
                                policy.actor.reset_state(batch_size=train_envs.env_num)
                                train_fn.high_loss_count = 0
                                
                                # Try loading previous checkpoint as last resort
                                try:
                                    latest_checkpoint = find_latest_checkpoint(args.save_dir)
                                    if latest_checkpoint:
                                        print(f"⏪ Rolling back to checkpoint {latest_checkpoint}")
                                        checkpoint = torch.load(latest_checkpoint)
                                        policy.load_state_dict(checkpoint['model'])
                                except Exception as e:
                                    print(f"Recovery failed: {e}")
                        else:
                            train_fn.high_loss_count = max(0, train_fn.high_loss_count - 1)
                    break  # Only need to process one loss metric
        
        # ===== 2. PARAMETER NAN CHECKING (PERIODIC) =====
        if env_step % 5000 == 0:
            # Check for NaN values in parameters
            nan_params = []
            for name, param in policy.named_parameters():
                if param.requires_grad and torch.isnan(param).any():
                    nan_params.append(name)
                    
            if nan_params:
                print(f"CRITICAL: NaN detected in parameters: {', '.join(nan_params)}")
                # Attempt recovery by loading previous checkpoint
                try:
                    latest_checkpoint = find_latest_checkpoint(args.save_dir)
                    if latest_checkpoint:
                        print(f"⚠️ Loading checkpoint {latest_checkpoint}")
                        checkpoint = torch.load(latest_checkpoint)
                        policy.load_state_dict(checkpoint['model'])
                except Exception as e:
                    print(f"❌ Recovery failed: {e}")
            
            # Basic reporting of spectral radius (monitoring only)
            if hasattr(policy.actor.net.recurrent, 'weight_hh'):
                with torch.no_grad():
                    try:
                        u, s, v = torch.linalg.svd(policy.actor.net.recurrent.weight_hh.detach())
                        spec_radius = s[0].item()
                        print(f"Spectral radius: {spec_radius:.4f}")
                    except Exception as e:
                        pass
    
    # Before starting training loop
    train_collector.reset()
    test_collector.reset()
    # Before test collection
    test_collector.reset_env()
    policy.actor.reset_state(batch_size=args.test_num)
    
    # Start training
    trainer = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epochs,
        step_per_epoch=args.step_per_epoch,
        repeat_per_collect=args.repeat_per_collect,
        episode_per_test=10,  # test episodes
        batch_size=args.batch_size,
        step_per_collect=args.step_per_collect,
        save_best_fn=save_best_fn,
        save_checkpoint_fn=save_checkpoint_fn,
        logger=logger,
        test_in_train=False,  # Test only at the end of each epoch
        train_fn=train_fn,
    )
    result = trainer.run()
    
    # Clean up
    train_envs.close()
    test_envs.close()
    

if __name__ == "__main__":
    main()