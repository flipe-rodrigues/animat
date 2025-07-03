import os
import torch
import numpy as np
from tianshou.utils import TensorboardLogger
from tianshou.trainer import onpolicy_trainer
from tianshou.data import Collector, VectorReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import argparse

from shimmy_wrapper import create_training_env, create_eval_env, create_env, set_seeds
from bio_policy_pg import make_rnn_actor


def get_args():
    parser = argparse.ArgumentParser()
    # Training parameters
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--train-num', type=int, default=16, help='number of training environments')
    parser.add_argument('--test-num', type=int, default=10, help='number of test environments')
    parser.add_argument('--buffer-size', type=int, default=50000, help='replay buffer size')
    parser.add_argument('--step-per-epoch', type=int, default=10000, help='steps per epoch')
    parser.add_argument('--step-per-collect', type=int, default=2000, help='steps per collection')
    parser.add_argument('--repeat-per-collect', type=int, default=1, help='repeat training per collection')
    
    # Network parameters
    parser.add_argument('--hidden-size', type=int, default=128, help='hidden layer size')
    parser.add_argument('--tau-mem', type=float, default=16.0, help='membrane time constant')
    
    # Logging and saving
    parser.add_argument('--log-dir', type=str, default='logs/pg_simple', help='log directory')
    parser.add_argument('--save-dir', type=str, default='models/pg_simple', help='model save directory')
    parser.add_argument('--save-interval', type=int, default=10, help='model save interval')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
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
    
    # Create policy
    policy = make_rnn_actor(
        obs_dim=state_shape[0],
        action_dim=action_shape[0],
        action_space=action_space,
        hidden_size=args.hidden_size,
        tau_mem=args.tau_mem,
        lr=args.lr,
        device=args.device
    )
    
    # Create buffer
    buffer = VectorReplayBuffer(
        args.buffer_size,
        len(train_envs)
    )

    # Create collectors
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
    
    # Simple save function
    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(args.save_dir, 'policy.pth'))
    
    # Reset states before training
    train_collector.reset()
    test_collector.reset()
    
    # Start training with minimal settings
    result = onpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epochs,
        step_per_epoch=args.step_per_epoch,
        repeat_per_collect=args.repeat_per_collect,
        episode_per_test=10,
        batch_size=args.batch_size,
        step_per_collect=args.step_per_collect,
        save_best_fn=save_best_fn,
        logger=logger,
        test_in_train=False
    )
    
    # Clean up
    train_envs.close()
    test_envs.close()
    
    print(f"Best reward: {result['best_reward']}")
    

if __name__ == "__main__":
    main()