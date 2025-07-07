import os
import torch
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import PPOPolicy
from tianshou.utils import TensorboardLogger
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.trainer import OnpolicyTrainer

from shimmy_wrapper import create_training_env_good, create_eval_env_good, set_seeds
from policy_networks_gru_fixed import GRUActorNetwork
from policy_networks import ModalitySpecificEncoder, CriticNetwork
from torch.distributions import Independent, Normal

update_step_counter = 0

def train_ppo(args):
    set_seeds(args.seed)
    
    log_path = os.path.join(args.logdir, 'ppo_gru_fixed')
    os.makedirs(log_path, exist_ok=True)
    
    absolute_log_path = os.path.abspath(log_path)
    print(f"=== TENSORBOARD LOGS LOCATION ===")
    print(f"Log directory: {absolute_log_path}")
    print(f"TensorBoard command: tensorboard --logdir={absolute_log_path}")
    print("="*50)
    
    writer = SummaryWriter(log_path)
    device = torch.device(args.device)
    
    # Create environments
    train_envs = create_training_env_good(num_envs=args.training_num, base_seed=args.seed)
    test_envs = create_eval_env_good(num_envs=args.test_num, base_seed=args.seed+100)
    
    obs_shape = train_envs.observation_space[0].shape
    action_shape = train_envs.action_space[0].shape
    
    print(f"Environment specs: obs_shape={obs_shape}, action_shape={action_shape}")
    
    # Create shared encoder
    shared_encoder = ModalitySpecificEncoder(target_size=40).to(device)
    
    # Create GRU actor
    actor = GRUActorNetwork(
        obs_shape=obs_shape,
        action_shape=action_shape,
        encoder=shared_encoder,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        device=device
    ).to(device)
    
    # Create critic - FIXED constructor call
    critic = CriticNetwork(
        obs_shape=obs_shape,
        encoder=shared_encoder,
        device=device
    ).to(device)

    # Debug network dimensions
    print(f"GRU Actor params: {sum(p.numel() for p in actor.parameters()):,}")
    print(f"Critic params: {sum(p.numel() for p in critic.parameters()):,}")
    print(f"Shared encoder params: {sum(p.numel() for p in shared_encoder.parameters()):,}")
    
    # Test forward pass to verify dimensions
    test_obs = torch.randn(1, *obs_shape).to(device)
    print(f"Test obs shape: {test_obs.shape}")
    
    # Test actor
    actor_state = actor.init_state(batch_size=1)
    (mu, sigma), new_state = actor(test_obs, state=actor_state)
    print(f"Actor output: mu shape={mu.shape}, sigma shape={sigma.shape}")
    print(f"Actor output ranges: mu=[{mu.min():.3f}, {mu.max():.3f}], sigma=[{sigma.min():.3f}, {sigma.max():.3f}]")
    
    # Test critic
    value = critic(test_obs)
    print(f"Critic output: value shape={value.shape}, value={value.item():.3f}")
    
    # Optimizer setup
    shared_params = list(shared_encoder.parameters())
    
    if len(shared_params) == 0:
        print("Using fixed shared encoder")
        all_params = set(actor.parameters()) | set(critic.parameters())
        optim = torch.optim.AdamW(
            all_params,
            lr=args.actor_lr,
            weight_decay=1e-4
        )
    else:
        print("Using trainable shared encoder")
        shared_param_ids = {id(p) for p in shared_encoder.parameters()}
        actor_specific_params = [p for p in actor.parameters() if id(p) not in shared_param_ids]
        critic_specific_params = [p for p in critic.parameters() if id(p) not in shared_param_ids]
        
        optim = torch.optim.AdamW([
            {'params': shared_params, 'lr': args.actor_lr, 'weight_decay': 1e-4},
            {'params': actor_specific_params, 'lr': args.actor_lr, 'weight_decay': 1e-4},
            {'params': critic_specific_params, 'lr': args.critic_lr, 'weight_decay': 1e-4},
        ])
    
    def dist(logits):
        mean, sigma = logits
        return Independent(Normal(mean, sigma), 1)
    
    # Create PPO policy
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
        action_space=train_envs.action_space[0],
        eps_clip=args.eps_clip,
        value_clip=args.value_clip,
        dual_clip=args.dual_clip,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
    )
    
    # Test policy forward pass
    from tianshou.data import Batch
    test_batch = Batch(obs=test_obs.cpu().numpy(), info={})
    policy_result = policy.forward(test_batch, state=None)
    print(f"Policy test: action shape={policy_result.act.shape}, action range=[{policy_result.act.min():.3f}, {policy_result.act.max():.3f}]")
    
    # Monitoring
    def monitor_training(policy, result=None, batch=None, env_step=None):
        global_step = env_step or update_step_counter
        
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

    original_update = policy.update

    def update_wrapper(*args, **kwargs):
        global update_step_counter
        result = original_update(*args, **kwargs)
        update_step_counter += 1
        monitor_training(policy, result=result, env_step=update_step_counter)
        return result

    policy.update = update_wrapper

    # Create buffer
    buffer = VectorReplayBuffer(
        total_size=args.buffer_size,
        buffer_num=args.training_num,
        stack_num=args.stack_num,
    )
    
    # Create collectors
    train_collector = Collector(policy, train_envs, buffer)
    test_collector = Collector(policy, test_envs)
    
    logger = TensorboardLogger(writer)
    
    # Training
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
        save_best_fn=lambda policy: torch.save(
            policy.state_dict(),
            os.path.join(log_path, 'policy_best.pth')
        ),
        save_checkpoint_fn=lambda epoch, env_step, gradient_step: torch.save(
            {
                'epoch': epoch,
                'env_step': env_step,
                'gradient_step': gradient_step,
                'policy': policy.state_dict(),
                'actor': policy.actor.state_dict(),
                'critic': policy.critic.state_dict(),
                'optim': optim.state_dict(),
            },
            os.path.join(log_path, f'checkpoint_{epoch}.pth')
        ) if epoch % args.save_interval == 0 else None,
        train_fn=lambda epoch, step: monitor_training(policy, env_step=step)
    ).run()
    
    torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))
    
    train_envs.close()
    test_envs.close()
    
    return result

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-num", type=int, default=16)
    parser.add_argument("--test-num", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--stack-num", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer-size", type=int, default=2048)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--step-per-epoch", type=int, default=2048)
    parser.add_argument("--step-per-collect", type=int, default=1024)
    parser.add_argument("--repeat-per-collect", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--eps-clip", type=float, default=0.2)
    parser.add_argument("--value-clip", type=int, default=1)
    parser.add_argument("--dual-clip", type=float, default=None)
    parser.add_argument("--norm-adv", type=int, default=1)
    parser.add_argument("--recompute-adv", type=int, default=1)
    parser.add_argument("--rew-norm", type=int, default=0)
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-interval", type=int, default=4)
    return parser

if __name__ == "__main__":
    args = make_parser().parse_args()
    result = train_ppo(args)
    print(f"Final reward: {result['best_reward']:.2f}")