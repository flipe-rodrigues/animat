"""Comprehensive test for all modular component combinations."""

import torch
import numpy as np
import time
from itertools import product

from encoders.encoders import IdentityEncoder, ModalitySpecificEncoder, GridEncoder
from networks.rnn import RNNPolicy
from envs.dm_env import make_arm_env
from wrappers.es_wrapper import ESRNNAdapter
from trainers.es_trainer import ModularESTrainer
from utils import encode_numpy, validate_compatibility

def test_all_encoder_combinations():
    """Test all possible encoder combinations."""
    print("🔄 TESTING ALL ENCODER COMBINATIONS")
    print("=" * 60)
    
    # Define encoder configurations
    encoder_configs = [
        ("Identity_15", IdentityEncoder, {"obs_dim": 15}),
        ("Grid_3x3", ModalitySpecificEncoder, {"grid_size": 3, "raw_obs_dim": 15}),
        ("Grid_5x5", ModalitySpecificEncoder, {"grid_size": 5, "raw_obs_dim": 15}),
        ("Grid_7x7", ModalitySpecificEncoder, {"grid_size": 7, "raw_obs_dim": 15}),
        ("Pure_Grid_3x3", GridEncoder, {"grid_size": 3}),
        ("Pure_Grid_5x5", GridEncoder, {"grid_size": 5}),
    ]
    
    # Define network configurations
    network_configs = [
        ("RNN_Small", RNNPolicy, {"hidden_size": 16, "alpha": 0.1}),
        ("RNN_Medium", RNNPolicy, {"hidden_size": 32, "alpha": 0.1}),
        ("RNN_Large", RNNPolicy, {"hidden_size": 64, "alpha": 0.1}),
    ]
    
    results = {}
    
    # Test all combinations
    for (enc_name, enc_class, enc_kwargs), (net_name, net_class, net_kwargs) in product(encoder_configs, network_configs):
        combo_name = f"{enc_name}_{net_name}"
        
        try:
            print(f"\n🧪 Testing {combo_name}...")
            
            # Create encoder
            encoder = enc_class(**enc_kwargs)
            
            # Skip incompatible combinations
            if enc_class == GridEncoder and net_class == RNNPolicy:
                # GridEncoder outputs 2D but we need action_dim for RNN
                print(f"   ⏭️  Skipping incompatible combination")
                results[combo_name] = {'status': 'skipped', 'reason': 'incompatible'}
                continue
            
            # Create network with compatible dimensions
            if net_class == RNNPolicy:
                network = net_class(
                    input_dim=encoder.output_dim,
                    action_dim=4,  # Fixed for our environment
                    **net_kwargs
                )
            else:
                network = net_class(
                    input_dim=encoder.output_dim,
                    **net_kwargs
                )
            
            # Validate compatibility
            validate_compatibility(encoder, network)
            
            # Test forward pass
            test_obs = torch.randn(15)  # Raw observation
            encoded_obs = encoder(test_obs)
            
            if isinstance(network, RNNPolicy):
                output, hidden = network(encoded_obs.unsqueeze(0))
                output = output.squeeze(0)
            else:
                output = network(encoded_obs)
            
            # Calculate parameter counts
            encoder_params = sum(p.numel() for p in encoder.parameters())
            network_params = sum(p.numel() for p in network.parameters())
            total_params = encoder_params + network_params
            
            print(f"   ✅ Success!")
            print(f"      Encoder: {encoder.input_dim} → {encoder.output_dim} ({encoder_params} params)")
            print(f"      Network: {network.input_dim} → {network.output_dim} ({network_params} params)")
            print(f"      Total: {total_params} parameters")
            
            results[combo_name] = {
                'status': 'success',
                'encoder_dim': encoder.output_dim,
                'encoder_params': encoder_params,
                'network_params': network_params,
                'total_params': total_params,
                'output_shape': output.shape
            }
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results[combo_name] = {'status': 'failed', 'error': str(e)}
    
    return results

def test_environment_integration():
    """Test integration with different environments and encoders."""
    print("\n🌍 TESTING ENVIRONMENT INTEGRATION")
    print("=" * 60)
    
    # Test different encoder-environment combinations
    test_configs = [
        ("Identity", IdentityEncoder(obs_dim=15)),
        ("Grid_Small", ModalitySpecificEncoder(grid_size=3, raw_obs_dim=15)),
        ("Grid_Large", ModalitySpecificEncoder(grid_size=7, raw_obs_dim=15)),
    ]
    
    results = {}
    
    for name, encoder in test_configs:
        try:
            print(f"\n🔬 Testing {name} with environment...")
            
            # Create environment
            env = make_arm_env(random_seed=42)
            timestep = env.reset()
            
            # Extract observation
            muscle_data = timestep.observation['muscle_sensors']
            target_pos = timestep.observation['target_position'].flatten()
            raw_obs = np.concatenate([muscle_data, target_pos])
            
            # Test encoding
            encoded_obs = encode_numpy(encoder, raw_obs)
            
            # Test multiple steps
            rewards = []
            observations = []
            encoded_observations = []
            
            for step in range(10):
                # Random action
                action = np.random.uniform(0, 1, size=4)
                timestep = env.step(action)
                
                # Extract and encode observation
                muscle_data = timestep.observation['muscle_sensors']
                target_pos = timestep.observation['target_position'].flatten()
                raw_obs = np.concatenate([muscle_data, target_pos])
                encoded_obs = encode_numpy(encoder, raw_obs)
                
                observations.append(raw_obs)
                encoded_observations.append(encoded_obs)
                rewards.append(timestep.reward)
                
                if timestep.last():
                    break
            
            observations = np.array(observations)
            encoded_observations = np.array(encoded_observations)
            avg_reward = np.mean(rewards)
            
            print(f"   ✅ Environment integration successful!")
            print(f"      Steps: {len(observations)}")
            print(f"      Raw obs shape: {observations.shape}")
            print(f"      Encoded obs shape: {encoded_observations.shape}")
            print(f"      Average reward: {avg_reward:.4f}")
            
            results[name] = {
                'status': 'success',
                'steps': len(observations),
                'raw_shape': observations.shape,
                'encoded_shape': encoded_observations.shape,
                'avg_reward': avg_reward,
                'encoder_output_dim': encoder.output_dim
            }
            
            env.close()
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results[name] = {'status': 'failed', 'error': str(e)}
    
    return results

def test_es_trainer_integration():
    """Test ES trainer with different configurations."""
    print("\n🔬 TESTING ES TRAINER INTEGRATION")
    print("=" * 60)
    
    configs = [
        ("ES_Identity_Small", "identity", 3, 16),
        ("ES_Grid_Small", "grid", 3, 16),
        ("ES_Grid_Medium", "grid", 5, 24),
    ]
    
    results = {}
    
    for name, encoder_type, grid_size, hidden_size in configs:
        try:
            print(f"\n⚡ Testing {name}...")
            
            # Create ES trainer
            trainer = ModularESTrainer(
                encoder_type=encoder_type,
                grid_size=grid_size,
                hidden_size=hidden_size,
                alpha=0.1
            )
            
            # Test single evaluation
            rnn = trainer.rnn_template.from_params(trainer.rnn_template.get_params())
            fitness = trainer.evaluate_rnn(rnn, seed=42, max_steps=50)
            
            # Test very short training
            print(f"      Running micro-training (2 generations)...")
            start_time = time.time()
            best_params, fitnesses = trainer.train(
                num_generations=2,
                sigma=0.5,
                save_dir=f"./test_results_{name.lower()}"
            )
            training_time = time.time() - start_time
            
            print(f"   ✅ ES training successful!")
            print(f"      Single evaluation fitness: {fitness:.4f}")
            print(f"      Training time: {training_time:.2f}s")
            print(f"      Total evaluations: {len(fitnesses)}")
            print(f"      Encoder output dim: {trainer.obs_dim}")
            print(f"      RNN parameters: {trainer.rnn_template.num_params}")
            
            results[name] = {
                'status': 'success',
                'single_fitness': fitness,
                'training_time': training_time,
                'total_evaluations': len(fitnesses),
                'encoder_dim': trainer.obs_dim,
                'rnn_params': trainer.rnn_template.num_params
            }
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {'status': 'failed', 'error': str(e)}
    
    return results

def performance_benchmark():
    """Benchmark performance of different configurations."""
    print("\n⚡ PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    configs = [
        ("Identity", IdentityEncoder(obs_dim=15)),
        ("Grid_3x3", ModalitySpecificEncoder(grid_size=3, raw_obs_dim=15)),
        ("Grid_5x5", ModalitySpecificEncoder(grid_size=5, raw_obs_dim=15)),
        ("Grid_7x7", ModalitySpecificEncoder(grid_size=7, raw_obs_dim=15)),
    ]
    
    results = {}
    num_samples = 1000
    
    for name, encoder in configs:
        print(f"\n🏃 Benchmarking {name}...")
        
        # Generate test data
        test_data = torch.randn(num_samples, 15)
        
        # Benchmark encoding
        start_time = time.time()
        with torch.no_grad():
            for obs in test_data:
                encoded = encoder(obs)
        encoding_time = time.time() - start_time
        
        # Memory usage (approximate)
        encoder_params = sum(p.numel() for p in encoder.parameters())
        
        print(f"   📊 Results:")
        print(f"      Encoding time: {encoding_time:.4f}s ({num_samples} samples)")
        print(f"      Time per sample: {encoding_time/num_samples*1000:.4f}ms")
        print(f"      Parameters: {encoder_params}")
        print(f"      Output dimension: {encoder.output_dim}")
        
        results[name] = {
            'encoding_time': encoding_time,
            'time_per_sample_ms': encoding_time/num_samples*1000,
            'parameters': encoder_params,
            'output_dim': encoder.output_dim
        }
    
    return results

if __name__ == "__main__":
    print("🚀 COMPREHENSIVE MODULAR SYSTEM TEST")
    print("=" * 80)
    
    # Run all tests
    print("\n" + "="*80)
    combo_results = test_all_encoder_combinations()
    
    print("\n" + "="*80)
    env_results = test_environment_integration()
    
    print("\n" + "="*80)
    es_results = test_es_trainer_integration()
    
    print("\n" + "="*80)
    perf_results = performance_benchmark()
    
    # Comprehensive summary
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE RESULTS SUMMARY")
    print("="*80)
    
    print("\n🔄 Encoder-Network Combinations:")
    successful_combos = 0
    total_combos = 0
    for name, result in combo_results.items():
        total_combos += 1
        if result['status'] == 'success':
            successful_combos += 1
            print(f"  ✅ {name}: {result['total_params']:,} params, {result['encoder_dim']}D encoding")
        elif result['status'] == 'skipped':
            print(f"  ⏭️  {name}: {result['reason']}")
        else:
            print(f"  ❌ {name}: {result['error']}")
    
    print(f"\n🌍 Environment Integration:")
    for name, result in env_results.items():
        if result['status'] == 'success':
            print(f"  ✅ {name}: {result['steps']} steps, {result['encoded_shape']} encoding")
        else:
            print(f"  ❌ {name}: {result['error']}")
    
    print(f"\n🔬 ES Trainer Integration:")
    for name, result in es_results.items():
        if result['status'] == 'success':
            print(f"  ✅ {name}: {result['rnn_params']} params, {result['total_evaluations']} evals")
        else:
            print(f"  ❌ {name}: {result['error']}")
    
    print(f"\n⚡ Performance Benchmark:")
    for name, result in perf_results.items():
        print(f"  📊 {name}: {result['time_per_sample_ms']:.3f}ms/sample, {result['output_dim']}D output")
    
    # Overall statistics
    total_tests = len(combo_results) + len(env_results) + len(es_results)
    passed_tests = (
        sum(1 for r in combo_results.values() if r['status'] == 'success') +
        sum(1 for r in env_results.values() if r['status'] == 'success') +
        sum(1 for r in es_results.values() if r['status'] == 'success')
    )
    
    print(f"\n🎉 FINAL RESULTS: {passed_tests}/{total_tests} tests passed")
    print(f"   Encoder combinations: {successful_combos}/{len(combo_results)} successful")
    
    if passed_tests == total_tests:
        print("\n🌟 ALL SYSTEMS OPERATIONAL!")
        print("Your modular architecture is fully functional and ready for research!")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} tests failed. Check the details above.")