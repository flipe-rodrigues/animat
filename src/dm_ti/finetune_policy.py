import os
import torch
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.utils import TensorboardLogger
# Fix the import to use the class instead of function
from tianshou.trainer import OnpolicyTrainer

from bio_policy_rnn import make_rnn_actor_critic
from shimmy_wrapper import create_training_env, create_eval_env, create_env, set_seeds

def get_finetune_args():
    # Create a new parser with all the same arguments as in train_bio.py
    parser = argparse.ArgumentParser()
    
    # Add all arguments manually (matching train_bio.py)
    # Training parameters
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--train-num', type=int, default=8, help='number of training environments')
    parser.add_argument('--test-num', type=int, default=10, help='number of test environments')
    parser.add_argument('--buffer-size', type=int, default=20000, help='replay buffer size')
    parser.add_argument('--step-per-epoch', type=int, default=5120, help='steps per epoch')
    parser.add_argument('--step-per-collect', type=int, default=1024, help='steps per collection')
    parser.add_argument('--repeat-per-collect', type=int, default=4, help='repeat training per collection')
    
    # Network parameters
    parser.add_argument('--hidden-size', type=int, default=128, help='hidden layer size')
    parser.add_argument('--tau-mem', type=float, default=8.0, help='membrane time constant')
    parser.add_argument('--tau-adapt', type=float, default=70.0, help='adaptation time constant')
    
    # Logging and saving
    parser.add_argument('--log-dir', type=str, default='logs', help='log directory')
    parser.add_argument('--save-dir', type=str, default='models', help='model save directory')
    parser.add_argument('--save-interval', type=int, default=10, help='model save interval')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Learning rate scheduler parameters
    parser.add_argument('--lr-decay', type=float, default=0.1, help='learning rate decay factor')
    parser.add_argument('--lr-decay-epochs', type=int, default=150, help='epochs before decay')
    parser.add_argument('--use-scheduler', action='store_true', help='use learning rate scheduler')
    
    # Add checkpoint path for fine-tuning
    parser.add_argument('--checkpoint-path', type=str, default='models/policy_good.pth', 
                        help='Path to load pretrained model')
    
    # Modify exploitation-specific parameters
    parser.set_defaults(
        # Exploitation settings
        ent_coef=0.02,         # Much lower entropy to exploit learned policy
        lr=1e-4,               # Lower learning rate for fine adjustments
        epochs=200,            # Keep 300 epochs for thorough fine-tuning
        train_num=16,           # Fewer environments for exploitation
        step_per_collect=512,  # Smaller batches for more focused updates
        
        # Checkpoint loading
        checkpoint_path='models/policy_good.pth',  # Path to best model
    )
    
    return parser.parse_args()

def main():
    args = get_finetune_args()
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set random seeds
    set_seeds(args.seed)
    
    # Create environments with fewer envs for exploitation
    train_envs = create_training_env(num_envs=args.train_num)
    
    # Create temporary environment to get shapes
    temp_env = create_env()
    state_shape = temp_env.observation_space.shape
    action_shape = temp_env.action_space.shape
    
    # Create a new policy with exploitation parameters
    policy = make_rnn_actor_critic(
        obs_dim=state_shape[0],
        action_dim=action_shape[0],
        action_space=temp_env.action_space,  # Add this line to provide the action space
        hidden_size=args.hidden_size,
        tau_mem=args.tau_mem,
        lr=args.lr,
        device=args.device,
    )
    
    # Load the weights from the best checkpoint
    policy.load_state_dict(torch.load(args.checkpoint_path))
    
    # Update the entropy coefficient directly
    policy._ent_coef = args.ent_coef
    
    # Create buffer
    buffer = VectorReplayBuffer(
        args.buffer_size,
        len(train_envs)
    )
    
    # Create collectors
    train_collector = Collector(policy, train_envs, buffer)
    
    # Create test environment for evaluation
    test_envs = create_eval_env(num_envs=10)
    test_collector = Collector(policy, test_envs)  # Collector works with vector environments too
    
    # Create logger
    writer = SummaryWriter(args.log_dir)
    logger = TensorboardLogger(writer)
    
    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(args.save_dir, 'policy_finetuned.pth'))
    
    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # Save checkpoint every save_interval epochs
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'model': policy.state_dict(),
                'optim': policy.optim.state_dict(),
                'epoch': epoch,
                'env_step': env_step,
                'gradient_step': gradient_step,
            }, os.path.join(args.save_dir, f"finetuned_checkpoint_{epoch}.pth"))
    
    # Before starting training loop
    train_collector.reset()
    test_collector.reset()
    
    # Start training with lower entropy for exploitation
    print(f"Starting fine-tuning with entropy coefficient: {args.ent_coef}")
    # Replace function call with class instantiation and run method
    trainer = OnpolicyTrainer(
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
        save_checkpoint_fn=save_checkpoint_fn,
        logger=logger,
        test_in_train=False,
    )
    result = trainer.run()
    
    # Clean up
    train_envs.close()
    test_envs.close()

if __name__ == "__main__":
    main()