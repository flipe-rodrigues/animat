import os
import torch
import numpy as np
from tianshou.utils import TensorboardLogger
from tianshou.trainer import OffpolicyTrainer  # Fixed import
from tianshou.data import VectorReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import argparse

from shimmy_wrapper import create_training_env, create_eval_env, create_env, set_seeds
from bio_policy_sac2 import create_simple_sac, create_collector, sync_hidden_states

def get_args():
    parser = argparse.ArgumentParser()
    # Training parameters
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--buffer-size', type=int, default=100000, help='replay buffer size')
    parser.add_argument('--hidden-size', type=int, default=64, help='hidden layer size')
    parser.add_argument('--tau-mem', type=float, default=8.0, help='membrane time constant')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--alpha', type=float, default=0.2, help='entropy coefficient')
    parser.add_argument('--auto-alpha', action='store_true', help='use auto entropy tuning')
    parser.add_argument('--batch-size', type=int, default=512, help='batch size')
    parser.add_argument('--step-per-epoch', type=int, default=5000, help='steps per epoch')
    parser.add_argument('--step-per-collect', type=int, default=512, help='steps per collection')
    parser.add_argument('--update-per-step', type=float, default=0.5, help='updates per step')
    parser.add_argument('--train-num', type=int, default=16, help='number of training environments')
    parser.add_argument('--test-num', type=int, default=5, help='number of test environments')
    
    # Logging and saving
    parser.add_argument('--log-dir', type=str, default='logs_rsac', help='log directory')
    parser.add_argument('--save-dir', type=str, default='models_rsac', help='model save directory')
    parser.add_argument('--save-interval', type=int, default=10, help='model save interval')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')
    
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
    obs_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.shape[0]
    action_space = temp_env.action_space
    
    # Create policy using our bio_policy_sac2
    policy = create_simple_sac(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_space=action_space,
        hidden_size=args.hidden_size,
        tau_mem=args.tau_mem,
        lr=args.lr,
        alpha=args.alpha,
        auto_alpha=args.auto_alpha,
        device=args.device,
        debug=args.debug
    )
    
    # Create buffer
    buffer = VectorReplayBuffer(
        args.buffer_size,
        len(train_envs)
    )
    
    # Use our custom collector that properly handles hidden states
    train_collector = create_collector(
        policy,
        train_envs,
        buffer,
        exploration_noise=True
    )
    test_collector = create_collector(
        policy,
        test_envs,
        buffer=None,
        exploration_noise=False
    )
    
    # Create logger
    writer = SummaryWriter(args.log_dir)
    logger = TensorboardLogger(writer)
    
    # Define callbacks for state management and model saving
    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(args.save_dir, 'policy.pth'))
    
    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # Save checkpoint every save_interval epochs
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'model': policy.state_dict(),
                'epoch': epoch,
                'env_step': env_step,
                'gradient_step': gradient_step,
            }, os.path.join(args.save_dir, f"checkpoint_{epoch}.pth"))
    
    # Function to reset hidden states only at the beginning of each training epoch
    def train_fn(epoch, env_step):
        if epoch == 0 and env_step == 0:  # Only reset at the very beginning
            print(f"Training start: Resetting hidden states")
            sync_hidden_states(policy)
        # Add learning rate scheduling here if needed
    
    # Function to reset hidden states before evaluation
    def test_fn(epoch, env_step):
        policy.eval()
        print(f"Evaluation at epoch {epoch}: Resetting hidden states")
        sync_hidden_states(policy)
    
    # Add monitoring for hidden states
    def log_state_norms(epoch, env_step):
        actor_state = policy.actor.shared.reset_hidden(1)
        critic1_state = policy.critic.get_hidden()
        critic2_state = policy.critic2.get_hidden()
        
        actor_norm = actor_state.norm().item() if actor_state is not None else 0
        critic1_norm = critic1_state.norm().item() if critic1_state is not None else 0
        critic2_norm = critic2_state.norm().item() if critic2_state is not None else 0
        
        logger.write("states/actor_norm", actor_norm, env_step)
        logger.write("states/critic1_norm", critic1_norm, env_step)
        logger.write("states/critic2_norm", critic2_norm, env_step)
    
    # Initialize collectors
    train_collector.reset()
    test_collector.reset()
    
    # Synchronize hidden states before starting
    sync_hidden_states(policy)
    
    # Start training
    trainer = OffpolicyTrainer(  # Use the class instead of function
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epochs,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=512,  # Make this a multiple of env_num (8)
        episode_per_test=5,
        batch_size=args.batch_size,
        update_per_step=args.update_per_step,
        train_fn=train_fn,
        test_fn=test_fn,
        save_best_fn=save_best_fn,
        save_checkpoint_fn=save_checkpoint_fn,
        logger=logger
    )
    result = trainer.run()  # Call run() on the trainer instance
    
    # Print results
    print(f"Finished training! Result: {result}")
    print(f"Final reward: {result['best_reward']}")
    
    # Clean up
    train_envs.close()
    test_envs.close()

if __name__ == "__main__":
    main()