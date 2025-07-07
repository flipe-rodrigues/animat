import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.utils import TensorboardLogger
from tianshou.policy import SACPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
import argparse

from shimmy_wrapper import create_training_env, create_eval_env, create_env, set_seeds


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--steps-per-epoch', type=int, default=5000, help='steps per epoch')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[256, 256], help='hidden layer sizes')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--tau', type=float, default=0.005, help='soft update coefficient')
    parser.add_argument('--alpha', type=float, default=0.2, help='entropy regularization coefficient')
    parser.add_argument('--auto-alpha', action='store_true', help='automatically tune alpha')
    parser.add_argument('--alpha-lr', type=float, default=3e-4, help='alpha learning rate')
    parser.add_argument('--start-timesteps', type=int, default=10000, help='timesteps for random exploration')
    parser.add_argument('--buffer-size', type=int, default=1000000, help='replay buffer size')
    parser.add_argument('--train-num', type=int, default=16, help='number of training environments')
    parser.add_argument('--test-num', type=int, default=5, help='number of test environments')
    parser.add_argument('--step-per-collect', type=int, default=64, help='steps per collection')
    parser.add_argument('--update-per-step', type=int, default=0.5, help='updates per step')
    parser.add_argument('--n-step', type=int, default=1, help='n-step return')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--log-dir', type=str, default='logs/sac', help='log directory')
    parser.add_argument('--save-dir', type=str, default='models/sac', help='model save directory')
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
    
    # Create temporary environment to get shapes
    temp_env = create_env()
    state_shape = temp_env.observation_space.shape[0]
    action_shape = temp_env.action_space.shape[0]
    max_action = temp_env.action_space.high[0]
    
    print(f"Observation shape: {state_shape}, Action shape: {action_shape}, Max action: {max_action}")
    
    # Create actor network
    net_a = Net(
        state_shape,
        hidden_sizes=args.hidden_sizes,
        activation=torch.nn.ReLU,
        device=args.device
    )
    actor = ActorProb(
        net_a,
        action_shape,
        max_action=max_action,
        unbounded=True,
        device=args.device
    )
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.lr)
    
    # Create critic networks (SAC uses two Q-networks)
    net_c1 = Net(
        state_shape,
        action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        activation=torch.nn.ReLU,
        device=args.device
    )
    critic1 = Critic(net_c1, device=args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.lr)
    
    net_c2 = Net(
        state_shape,
        action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        activation=torch.nn.ReLU,
        device=args.device
    )
    critic2 = Critic(net_c2, device=args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.lr)
    
    # Auto-alpha logic
    if args.auto_alpha:
        target_entropy = -np.prod(action_shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)
    
    # Create SAC policy
    policy = SACPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic1=critic1,           # Changed from critic
        critic1_optim=critic1_optim,  # Changed from critic_optim
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        estimation_step=args.n_step,
        action_space=temp_env.action_space
    )
    
    # Create replay buffer - SAC is off-policy, so we use a large buffer
    buffer = VectorReplayBuffer(
        args.buffer_size,
        len(train_envs)
    )
    
    # Create collectors
    train_collector = Collector(policy, train_envs, buffer)
    test_collector = Collector(policy, test_envs)
    
    # Create logger
    writer = SummaryWriter(args.log_dir)
    logger = TensorboardLogger(writer)
    
    # Define save functions with distinct names
    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(args.save_dir, 'sac_best.pth'))

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        if (epoch + 1) % 10 == 0:
            torch.save({
                'model': policy.state_dict(),
                'actor_optim': policy.actor_optim.state_dict(),
                'critic_optim': policy.critic_optim.state_dict(),
                'critic2_optim': policy.critic2_optim.state_dict(),
                'epoch': epoch,
                'env_step': env_step,
                'gradient_step': gradient_step,
            }, os.path.join(args.save_dir, f"sac_checkpoint_{epoch}.pth"))
    
    # Initial exploration
    if args.start_timesteps > 0:
        print(f"Initial exploration: collecting {args.start_timesteps} timesteps with random actions")
        train_collector.collect(n_step=args.start_timesteps, random=True)
    
    # Reset collectors
    train_collector.reset()
    test_collector.reset()
    
    # Start training
    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epochs,
        step_per_epoch=args.steps_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=10,
        batch_size=args.batch_size,
        save_best_fn=save_best_fn,
        save_checkpoint_fn=save_checkpoint_fn,
        update_per_step=args.update_per_step,
        logger=logger,
        test_in_train=False
    )
    
    print(f"Finished training! Best reward: {result['best_reward']}")
    
    # Clean up
    train_envs.close()
    test_envs.close()


if __name__ == "__main__":
    main()