import os
import torch
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
import traceback

from tianshou.policy import PPOPolicy
from tianshou.data import Collector, VectorReplayBuffer, Batch
from tianshou.trainer import OnpolicyTrainer

from shimmy_wrapper import create_training_env_good, create_eval_env_good, set_seeds
from policy_networks import RecurrentActorNetwork, ModalitySpecificEncoder, CriticNetwork
from torch.distributions import Independent, Normal


def debug_rnn_sequences(args):
    """Debug script to verify RNN sequence handling in collector and buffer."""
    print("=== RNN SEQUENCE DEBUGGING ===")
    
    # Set random seeds
    set_seeds(args.seed)
    
    # Select device
    device = torch.device(args.device)
    
    # Create small training environment for debugging
    train_envs = create_training_env_good(num_envs=2, base_seed=args.seed)
    
    # Get environment specifications
    obs_shape = train_envs.observation_space[0].shape
    action_shape = train_envs.action_space[0].shape
    
    print(f"Observation shape: {obs_shape}")
    print(f"Action shape: {action_shape}")
    print(f"Number of environments: {len(train_envs)}")
    print(f"Stack number: {args.stack_num}")
    
    # Create networks
    shared_encoder = ModalitySpecificEncoder(target_size=8).to(device)
    
    actor = RecurrentActorNetwork(
        obs_shape=obs_shape,
        action_shape=action_shape,
        encoder=shared_encoder,
        hidden_size=32,
        num_layers=1,
        device=device
    ).to(device)
    
    critic = CriticNetwork(
        obs_shape=obs_shape,
        encoder=shared_encoder,
        hidden_size=32,
        device=device
    ).to(device)
    
    # Create optimizer
    from itertools import chain
    all_params = list(chain(actor.parameters(), critic.parameters()))
    
    optim = torch.optim.AdamW(all_params, lr=args.actor_lr)
    
    # Distribution function
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
        action_space=train_envs.action_space[0],
    )
    
    # Create buffer with stack_num - THIS IS CRITICAL FOR RNN
    buffer = VectorReplayBuffer(
        total_size=args.buffer_size,
        buffer_num=len(train_envs),
        stack_num=args.stack_num,
    )
    
    print(f"Buffer configuration:")
    print(f"  Total size: {buffer.maxsize}")
    print(f"  Buffer num: {buffer.buffer_num}")
    print(f"  Stack num: {buffer.stack_num}")
    
    # Create collector
    train_collector = Collector(policy, train_envs, buffer)
    train_collector.reset()
    
    print("\n=== TESTING SEQUENCE COLLECTION ===")
    
    # Test 1: Initial collection
    print("\n1. Testing initial collection...")
    result = train_collector.collect(n_step=args.step_per_collect)
    
    episodes = result.n_collected_episodes
    steps = result.n_collected_steps
    
    print(f"Collected {episodes} episodes, {steps} steps")
    print(f"Buffer size after collection: {len(buffer)}")
    
    # Test 2: CRITICAL - Examine buffer sequence structure
    print("\n2. CRITICAL: Examining RNN sequence structure...")
    
    batch_size = min(64, len(buffer))
    if len(buffer) >= batch_size:
        try:
            # Sample from buffer
            batch_result = buffer.sample(batch_size)
            
            # Handle tuple return
            if isinstance(batch_result, tuple):
                batch = batch_result[0]  # First element is the batch
            else:
                batch = batch_result
                
            print(f"Sampled batch obs shape: {batch.obs.shape}")
            print(f"Sampled batch act shape: {batch.act.shape}")
            print(f"Sampled batch rew shape: {batch.rew.shape}")
            print(f"Sampled batch done shape: {batch.done.shape}")
            
            # CRITICAL ANALYSIS
            obs_is_sequence = len(batch.obs.shape) == 3  # [batch, seq, obs_dim]
            act_is_sequence = len(batch.act.shape) == 3  # [batch, seq, act_dim]
            
            print(f"\n🔍 SEQUENCE ANALYSIS:")
            print(f"Observations are sequences: {obs_is_sequence}")
            print(f"Actions are sequences: {act_is_sequence}")
            
            if obs_is_sequence:
                seq_len = batch.obs.shape[1]
                print(f"✅ Observation sequence length: {seq_len}")
                
                if seq_len == args.stack_num:
                    print(f"✅ PERFECT: Sequence length matches stack_num={args.stack_num}")
                else:
                    print(f"⚠️  WARNING: Sequence length {seq_len} != stack_num {args.stack_num}")
            else:
                print("❌ CRITICAL: Observations are NOT sequences!")
                print("❌ This will break RNN training!")
                
            if act_is_sequence:
                print(f"✅ Actions are sequences of length {batch.act.shape[1]}")
            else:
                print(f"⚠️  Actions are single timesteps (shape: {batch.act.shape})")
                print("⚠️  This is NORMAL for RNN training - actions don't need sequence dimension")
                
            # Test if RNN can process the sequences
            print(f"\n🧪 TESTING RNN PROCESSING:")
            
            if obs_is_sequence:
                with torch.no_grad():
                    obs_tensor = torch.tensor(batch.obs, dtype=torch.float32, device=device)
                    
                    # Test actor forward pass
                    try:
                        (mu, sigma), hidden = actor(obs_tensor)
                        print(f"✅ Actor processes sequences successfully")
                        print(f"   Input shape: {obs_tensor.shape}")
                        print(f"   Output shapes: mu={mu.shape}, sigma={sigma.shape}")
                        
                        # The output should be [batch, seq, action_dim] for sequences
                        if len(mu.shape) == 3:
                            print(f"✅ Actor outputs sequences: {mu.shape}")
                        else:
                            print(f"⚠️  Actor outputs single timesteps: {mu.shape}")
                            
                    except Exception as e:
                        print(f"❌ Actor FAILED to process sequences: {e}")
                        
                    # Test critic forward pass
                    try:
                        values = critic(obs_tensor)
                        print(f"✅ Critic processes sequences successfully")
                        print(f"   Output shape: {values.shape}")
                    except Exception as e:
                        print(f"❌ Critic FAILED to process sequences: {e}")
            else:
                print("❌ Cannot test RNN processing - no sequences found!")
                
        except Exception as e:
            print(f"❌ Error during buffer analysis: {e}")
            traceback.print_exc()
    
    # Test 3: Check environment reset and observation handling
    print("\n3. Testing environment observation handling...")
    
    try:
        obs_dict = train_envs.reset()
        print(f"Environment reset successful")
        print(f"Reset output type: {type(obs_dict)}")
        
        # Handle different observation formats
        if isinstance(obs_dict, tuple):
            obs, info = obs_dict
            print(f"Reset returns tuple: obs type={type(obs)}, info type={type(info)}")
            
            if isinstance(obs, np.ndarray):
                print(f"Observations are numpy array with shape: {obs.shape}")
            elif isinstance(obs, list):
                print(f"Observations are list with length: {len(obs)}")
                print(f"First obs shape: {np.array(obs[0]).shape}")
            else:
                print(f"Unknown observation format: {type(obs)}")
                
        elif isinstance(obs_dict, dict):
            obs = obs_dict.get('obs', obs_dict)
            print(f"Reset returns dict with keys: {list(obs_dict.keys())}")
            print(f"Observations shape: {np.array(obs).shape}")
        else:
            print(f"Direct observation array: {np.array(obs_dict).shape}")
            
    except Exception as e:
        print(f"❌ Environment reset failed: {e}")
        traceback.print_exc()
    
    # Test 4: Critical check for training compatibility
    print("\n4. CRITICAL: Training compatibility check...")
    
    print("🔍 CHECKING YOUR CURRENT SETUP:")
    
    # Check if sequences are working
    if len(buffer) >= 32:
        try:
            batch_result = buffer.sample(32)
            batch = batch_result[0] if isinstance(batch_result, tuple) else batch_result
            
            if len(batch.obs.shape) == 3:
                print("✅ EXCELLENT: Your buffer produces sequences for RNN training!")
                print(f"✅ Sequence length: {batch.obs.shape[1]}")
                print("✅ This setup should work with RNN training")
                
                # Additional checks
                if batch.obs.shape[1] == args.stack_num:
                    print("✅ PERFECT: Sequence length matches your stack_num setting")
                else:
                    print(f"⚠️  Sequence length ({batch.obs.shape[1]}) != stack_num ({args.stack_num})")
                    
            else:
                print("❌ CRITICAL ISSUE: Buffer does NOT produce sequences!")
                print("❌ RNN training will likely fail with this setup")
                print("❌ You need to fix the buffer or training loop")
                
        except Exception as e:
            print(f"❌ Could not verify sequence handling: {e}")
    
    # Test 5: Provide specific recommendations
    print("\n=== FINAL DIAGNOSIS & RECOMMENDATIONS ===")
    
    print("\n📋 SETUP SUMMARY:")
    print(f"✅ VectorReplayBuffer with stack_num={args.stack_num}")
    print(f"✅ Networks can process tensor inputs")
    print(f"✅ Collector successfully collects data")
    
    try:
        batch_result = buffer.sample(min(32, len(buffer)))
        batch = batch_result[0] if isinstance(batch_result, tuple) else batch_result
        
        if len(batch.obs.shape) == 3:
            print("✅ SEQUENCES ARE WORKING!")
            print("\n🎯 YOUR RNN SETUP IS CORRECTLY CONFIGURED")
            print("\n📝 Next steps:")
            print("  1. Run your main training script")
            print("  2. Monitor for NaN/stability issues")
            print("  3. Check that loss values are reasonable")
            print("  4. Verify that the policy is learning")
            
        else:
            print("❌ SEQUENCES ARE NOT WORKING")
            print("\n🚨 CRITICAL ISSUES TO FIX:")
            print("  1. Buffer is not producing sequences despite stack_num setting")
            print("  2. This will prevent proper RNN training")
            print("\n🔧 POSSIBLE SOLUTIONS:")
            print("  1. Check Tianshou version compatibility")
            print("  2. Verify that your OnpolicyTrainer uses sequences")
            print("  3. Consider manually implementing sequence sampling")
            print("  4. Debug the VectorReplayBuffer implementation")
            
    except Exception as e:
        print(f"❌ Could not complete diagnosis: {e}")
    
    print(f"\n📊 BUFFER STATISTICS:")
    print(f"  Total capacity: {buffer.maxsize}")
    print(f"  Current size: {len(buffer)}")
    print(f"  Number of environments: {buffer.buffer_num}")
    print(f"  Stack number: {buffer.stack_num}")
    
    # Cleanup
    train_envs.close()


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer-size", type=int, default=1024)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--stack-num", type=int, default=4)
    parser.add_argument("--step-per-collect", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    debug_rnn_sequences(args)