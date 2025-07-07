import gymnasium as gym
import torch

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.discrete import Actor, Critic

device = "cuda" if torch.cuda.is_available() else "cpu"

# environments
env = gym.make("CartPole-v1")
train_envs = DummyVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(20)])
test_envs = DummyVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(10)])

# model & optimizer
assert env.observation_space.shape is not None  # for mypy
net = Net(state_shape=env.observation_space.shape, hidden_sizes=[64, 64], device=device)

assert isinstance(env.action_space, gym.spaces.Discrete)  # for mypy
actor = Actor(preprocess_net=net, action_shape=env.action_space.n, device=device).to(device)
critic = Critic(preprocess_net=net, device=device).to(device)
actor_critic = ActorCritic(actor, critic)
optim = torch.optim.Adam(actor_critic.parameters(), lr=0.0003)

# PPO policy
dist = torch.distributions.Categorical
policy: PPOPolicy = PPOPolicy(
    actor=actor,
    critic=critic,
    optim=optim,
    dist_fn=dist,
    action_space=env.action_space,
    action_scaling=False,
)

# collector
train_collector = Collector(policy, train_envs, VectorReplayBuffer(20000, len(train_envs)))
test_collector = Collector(policy, test_envs)

# trainer
train_result = OnpolicyTrainer(
    policy=policy,
    batch_size=256,
    train_collector=train_collector,
    test_collector=test_collector,
    max_epoch=10,
    step_per_epoch=50000,
    repeat_per_collect=10,
    episode_per_test=10,
    step_per_collect=2000,
    stop_fn=lambda mean_reward: mean_reward >= 195,
).run()

train_result.pprint_asdict()

# Let's watch its performance!
policy.eval()
eval_result = test_collector.collect(n_episode=3, render=False)
print(f"Final reward: {eval_result.returns.mean()}, length: {eval_result.lens.mean()}")