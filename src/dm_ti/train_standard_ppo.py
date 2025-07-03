import os
import torch
import numpy as np
import tianshou as ts
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import Net, ActorCritic
from tianshou.utils.net.continuous import ActorProb, Critic
from torch.optim.lr_scheduler import LambdaLR
import argparse
from torch import distributions as torch_dist

from shimmy_wrapper import create_training_env, create_eval_env, create_env, set_seeds
from networks import ModalitySpecificEncoder

class EncodingWrapper(torch.nn.Module):
    def __init__(self, base_net, encoder):
        super().__init__()
        self.base_net = base_net
        self.encoder = encoder
    
    def forward(self, obs, **kwargs):
        if isinstance(obs, dict):
            x = obs.get('obs', None)
        else:
            x = obs
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=self.encoder.device)
        encoded_obs = self.encoder(x)
        return self.base_net(encoded_obs, **kwargs)

class NormalizedEncodingWrapper(torch.nn.Module):
    def __init__(self, base_net, encoder, normalizer):
        super().__init__()
        self.base_net = base_net
        self.encoder = encoder
        self.normalizer = normalizer

    def forward(self, obs, **kwargs):
        if isinstance(obs, dict):
            x = obs.get('obs', None)
        else:
            x = obs
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=self.encoder.device)
        encoded_obs = self.encoder(x)
        normalized_obs = self.normalizer(encoded_obs)
        return self.base_net(normalized_obs, **kwargs)

    @property
    def max_action(self):
        return getattr(self.base_net, "max_action", None)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--lr-decay', action='store_true', help='enable learning rate decay')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--train-num', type=int, default=20, help='number of training environments')
    parser.add_argument('--test-num', type=int, default=10, help='number of test environments')
    parser.add_argument('--step-per-epoch', type=int, default=50000, help='steps per epoch')
    parser.add_argument('--step-per-collect', type=int, default=2000, help='steps per collection')
    parser.add_argument('--repeat-per-collect', type=int, default=10, help='repeat times per collection')
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[64, 64], help='hidden layer sizes')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--log-dir', type=str, default='logs/encoded_ppo', help='log directory')
    parser.add_argument('--save-dir', type=str, default='models/encoded_ppo', help='model save directory')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda')
    return parser.parse_args()

def main():
    args = get_args()
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    set_seeds(args.seed)
    train_envs = create_training_env(num_envs=args.train_num)
    test_envs = create_eval_env(num_envs=args.test_num)
    temp_env = create_env()
    state_shape = temp_env.observation_space.shape[0]
    action_shape = temp_env.action_space.shape[0]
    max_action = temp_env.action_space.high[0]

    encoder = ModalitySpecificEncoder(
        target_size=40,
        device=args.device
    )
    encoded_state_shape = encoder.output_size  # This should be 93
    
    print(f"Raw observation shape: {state_shape}")  # 15
    print(f"Encoded observation shape: {encoded_state_shape}")  # 93
    
    # Build networks for ENCODED observations
    net_a = Net(
        encoded_state_shape,  # Use 93D, not 15D
        hidden_sizes=args.hidden_sizes,
        activation=torch.nn.Tanh,
        device=args.device,
    )
    actor = ActorProb(
        net_a,
        action_shape,
        unbounded=True,
        device=args.device,
    ).to(args.device)
    
    net_c = Net(
        encoded_state_shape,  # Use 93D, not 15D
        hidden_sizes=args.hidden_sizes,
        activation=torch.nn.Tanh,
        device=args.device,
    )
    critic = Critic(net_c, device=args.device).to(args.device)
    
    actor_critic = ActorCritic(actor, critic)
    
    # Now the wrapper makes sense - it converts 15D→93D for networks expecting 93D
    encoded_actor_critic = EncodingWrapper(actor_critic, encoder)

    # Network initialization
    if hasattr(actor, 'sigma_param'):
        torch.nn.init.constant_(actor.sigma_param, -0.5)
    
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            # Orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
    
    # Policy layer scaling for last actor layer
    for m in actor.mu.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)

    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    lr_scheduler = None
    if args.lr_decay:
        max_update_num = np.ceil(args.step_per_epoch / args.step_per_collect) * args.epochs
        lr_scheduler = LambdaLR(optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num)

    policy = PPOPolicy(
        actor=actor,  # Use original actor, not wrapped
        critic=critic,  # Use original critic, not wrapped
        optim=optim,
        lr_scheduler=lr_scheduler,  # Add this line
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,  # Add this line
        eps_clip=0.2,
        vf_coef=0.5,
        ent_coef=0.0,
        action_space=temp_env.action_space,
        action_scaling=True,
        reward_normalization=False,
        advantage_normalization=False,
        recompute_advantage=True,
        value_clip=False,
        max_grad_norm=0.5,
        dist_fn=lambda x: torch.distributions.Independent(
            torch.distributions.Normal(*x), 1
        )
    )

    buffer = VectorReplayBuffer(
        args.step_per_collect * 2, 
        len(train_envs)
    )
    train_collector = Collector(policy, train_envs, buffer)
    test_collector = Collector(policy, test_envs)

    writer = SummaryWriter(args.log_dir)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(args.save_dir, 'encoded_ppo_best.pth'))

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        if (epoch + 1) % 10 == 0:
            torch.save({
                'model': policy.state_dict(),
                'optim': policy.optim.state_dict(),
            }, os.path.join(args.save_dir, f"encoded_ppo_checkpoint_{epoch}.pth"))

    train_collector.reset()
    test_collector.reset()

    trainer = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epochs,
        step_per_epoch=args.step_per_epoch,
        repeat_per_collect=args.repeat_per_collect,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        step_per_collect=args.step_per_collect,
        save_best_fn=save_best_fn,
        save_checkpoint_fn=save_checkpoint_fn,
        logger=logger,
        test_in_train=False,
    )

    result = trainer.run()
    print(f"Finished training! Best reward: {result['best_reward']}")
    train_envs.close()
    test_envs.close()

if __name__ == "__main__":
    main()