#!/usr/bin/env python3
"""
Comprehensive test script for Evolution Strategies (ES) training approach.
Tests all components and runs mini training sessions.
"""

import sys
import os
import time
import numpy as np
import torch
from pathlib import Path

# Add workspace to path
workspace_root = Path(__file__).parent
sys.path.insert(0, str(workspace_root))

def test_es_components():
    """Test individual ES components."""
    print("🔧 TESTING ES COMPONENTS")
    print("=" * 50)
    
    results = {}
    
    # Test 1: Environment
    try:
        print("\n1. Testing Environment...")
        from envs.dm_env import make_arm_env
        
        env = make_arm_env(random_seed=42)
        timestep = env.reset()
        
        print(f"   ✅ Environment created successfully")
        print(f"   📊 Observation keys: {list(timestep.observation.keys())}")
        
        # Test action
        action = np.random.uniform(0, 1, size=4)  # 4 muscle activations
        timestep = env.step(action)
        print(f"   ✅ Environment step successful, reward: {timestep.reward:.4f}")
        
        env.close()
        results['environment'] = {'status': 'success'}
        
    except Exception as e:
        print(f"   ❌ Environment test failed: {e}")
        results['environment'] = {'status': 'failed', 'error': str(e)}
    
    # Test 2: Encoders
    try:
        print("\n2. Testing Encoders...")
        from encoders.encoders import IdentityEncoder, ModalitySpecificEncoder
        
        # Test identity encoder
        identity_encoder = IdentityEncoder(obs_dim=15)
        test_obs = np.random.randn(15)
        encoded = identity_encoder(torch.FloatTensor(test_obs))
        print(f"   ✅ Identity Encoder: {test_obs.shape} → {encoded.shape}")
        
        # Test grid encoder
        grid_encoder = ModalitySpecificEncoder(grid_size=5, raw_obs_dim=15)
        encoded_grid = grid_encoder(torch.FloatTensor(test_obs))
        print(f"   ✅ Grid Encoder: {test_obs.shape} → {encoded_grid.shape}")
        
        results['encoders'] = {'status': 'success'}
        
    except Exception as e:
        print(f"   ❌ Encoder test failed: {e}")
        results['encoders'] = {'status': 'failed', 'error': str(e)}
    
    # Test 3: ES RNN Adapter
    try:
        print("\n3. Testing ES RNN Adapter...")
        from wrappers.es_wrapper import ESRNNAdapter
        
        adapter = ESRNNAdapter(input_dim=15, action_dim=4, hidden_size=25)
        print(f"   ✅ ES Adapter created: {adapter.num_params} parameters")
        
        # Test parameter operations
        params = adapter.get_params()
        print(f"   ✅ Parameter extraction: shape {params.shape}")
        
        new_adapter = adapter.from_params(params)
        print(f"   ✅ Parameter setting successful")
        
        # Test forward pass
        obs = np.random.randn(15)
        adapter.init_state()
        action = adapter.step(obs)
        print(f"   ✅ Forward pass: {obs.shape} → {action.shape}")
        print(f"   📊 Action range: [{action.min():.3f}, {action.max():.3f}]")
        
        results['es_adapter'] = {'status': 'success', 'params': adapter.num_params}
        
    except Exception as e:
        print(f"   ❌ ES Adapter test failed: {e}")
        results['es_adapter'] = {'status': 'failed', 'error': str(e)}
    
    # Test 4: ES Trainer
    try:
        print("\n4. Testing ES Trainer...")
        from trainers.es_trainer import ModularESTrainer
        
        trainer = ModularESTrainer(
            encoder_type='identity',
            hidden_size=16  # Small for testing
        )
        print(f"   ✅ ES Trainer created")
        print(f"   📊 Encoder: {trainer.encoder.__class__.__name__}")
        print(f"   📊 Observation dim: {trainer.obs_dim}")
        print(f"   📊 Action dim: {trainer.action_dim}")
        print(f"   📊 RNN parameters: {trainer.rnn_template.num_params}")
        
        results['es_trainer'] = {
            'status': 'success',
            'obs_dim': trainer.obs_dim,
            'action_dim': trainer.action_dim,
            'params': trainer.rnn_template.num_params
        }
        
    except Exception as e:
        print(f"   ❌ ES Trainer test failed: {e}")
        results['es_trainer'] = {'status': 'failed', 'error': str(e)}
    
    return results

def test_es_evaluation():
    """Test ES evaluation function."""
    print("\n🎯 TESTING ES EVALUATION")
    print("=" * 50)
    
    results = {}
    
    encoder_configs = [
        ('identity', {}),
        ('grid', {'grid_size': 3}),
        ('grid', {'grid_size': 5}),
    ]
    
    for encoder_type, kwargs in encoder_configs:
        config_name = f"{encoder_type}_{kwargs.get('grid_size', 'default')}"
        
        try:
            print(f"\n🔬 Testing {config_name}...")
            
            from trainers.es_trainer import ModularESTrainer
            
            trainer = ModularESTrainer(
                encoder_type=encoder_type,
                hidden_size=16,  # Small for speed
                **kwargs
            )
            
            # Test single evaluation
            rnn = trainer.rnn_template.from_params(trainer.rnn_template.get_params())
            
            start_time = time.time()
            fitness = trainer.evaluate_rnn(rnn, seed=42, max_steps=50)
            eval_time = time.time() - start_time
            
            print(f"   ✅ Evaluation successful!")
            print(f"   📊 Fitness: {fitness:.4f}")
            print(f"   ⏱️  Time: {eval_time:.3f}s")
            print(f"   🧠 Encoder output: {trainer.obs_dim}D")
            
            # Test multiple evaluations for consistency
            fitnesses = []
            for seed in range(3):
                f = trainer.evaluate_rnn(rnn, seed=seed, max_steps=30)
                fitnesses.append(f)
            
            print(f"   📈 Multi-eval consistency: {np.mean(fitnesses):.4f} ± {np.std(fitnesses):.4f}")
            
            results[config_name] = {
                'status': 'success',
                'fitness': fitness,
                'eval_time': eval_time,
                'obs_dim': trainer.obs_dim,
                'consistency': np.std(fitnesses)
            }
            
        except Exception as e:
            print(f"   ❌ {config_name} failed: {e}")
            results[config_name] = {'status': 'failed', 'error': str(e)}
    
    return results

def test_es_mini_training():
    """Test mini ES training sessions."""
    print("\n🏃 TESTING MINI ES TRAINING")
    print("=" * 50)
    
    results = {}
    
    training_configs = [
        ('identity_tiny', 'identity', {}, 8, 2),
        ('grid_tiny', 'grid', {'grid_size': 3}, 8, 2),
    ]
    
    for name, encoder_type, encoder_kwargs, hidden_size, generations in training_configs:
        try:
            print(f"\n⚡ Testing {name} training...")
            
            from trainers.es_trainer import ModularESTrainer
            
            trainer = ModularESTrainer(
                encoder_type=encoder_type,
                hidden_size=hidden_size,
                **encoder_kwargs
            )
            
            print(f"   🚀 Starting {generations}-generation training...")
            print(f"   📊 Parameters to optimize: {trainer.rnn_template.num_params}")
            
            start_time = time.time()
            best_params, fitnesses = trainer.train(
                num_generations=generations,
                sigma=0.5,
                save_dir=f"./test_es_{name}"
            )
            training_time = time.time() - start_time
            
            print(f"   ✅ Training completed!")
            print(f"   ⏱️  Training time: {training_time:.2f}s")
            print(f"   📈 Total evaluations: {len(fitnesses)}")
            
            if fitnesses:
                final_fitness = fitnesses[-1][2]  # (gen, ind, fitness)
                print(f"   🎯 Final fitness: {final_fitness:.4f}")
            
            # Test loading and evaluating best model
            if best_params is not None:
                best_rnn = trainer.rnn_template.from_params(best_params)
                eval_fitness = trainer.evaluate_rnn(best_rnn, seed=0, max_steps=100)
                print(f"   🔍 Best model evaluation: {eval_fitness:.4f}")
            
            results[name] = {
                'status': 'success',
                'training_time': training_time,
                'total_evaluations': len(fitnesses),
                'final_fitness': final_fitness if fitnesses else 0,
                'eval_fitness': eval_fitness if best_params is not None else 0,
                'encoder_dim': trainer.obs_dim
            }
            
        except Exception as e:
            print(f"   ❌ {name} training failed: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {'status': 'failed', 'error': str(e)}
    
    return results

def test_es_model_operations():
    """Test model saving and loading."""
    print("\n💾 TESTING MODEL SAVE/LOAD")
    print("=" * 50)
    
    try:
        from trainers.es_trainer import ModularESTrainer
        
        # Create trainer and train briefly
        trainer = ModularESTrainer(encoder_type='identity', hidden_size=8)
        
        # Get initial model
        initial_rnn = trainer.rnn_template.from_params(trainer.rnn_template.get_params())
        initial_fitness = trainer.evaluate_rnn(initial_rnn, seed=42, max_steps=30)
        
        # Save model
        save_path = "./test_model.pkl"
        trainer.save_model(initial_rnn, save_path)
        print(f"   ✅ Model saved to {save_path}")
        
        # Load model
        loaded_rnn = trainer.load_model(save_path)
        loaded_fitness = trainer.evaluate_rnn(loaded_rnn, seed=42, max_steps=30)
        print(f"   ✅ Model loaded successfully")
        
        # Check consistency
        fitness_diff = abs(initial_fitness - loaded_fitness)
        print(f"   📊 Initial fitness: {initial_fitness:.4f}")
        print(f"   📊 Loaded fitness: {loaded_fitness:.4f}")
        print(f"   📊 Difference: {fitness_diff:.6f}")
        
        if fitness_diff < 1e-6:
            print(f"   ✅ Save/load consistency verified!")
            status = 'success'
        else:
            print(f"   ⚠️  Small inconsistency detected")
            status = 'warning'
        
        # Test evaluation function
        fitnesses = trainer.evaluate_best(save_path, num_episodes=3)
        print(f"   📈 Multi-episode evaluation: {np.mean(fitnesses):.4f} ± {np.std(fitnesses):.4f}")
        
        # Cleanup
        if os.path.exists(save_path):
            os.remove(save_path)
        
        return {
            'status': status,
            'fitness_diff': fitness_diff,
            'mean_fitness': np.mean(fitnesses),
            'std_fitness': np.std(fitnesses)
        }
        
    except Exception as e:
        print(f"   ❌ Model operations test failed: {e}")
        return {'status': 'failed', 'error': str(e)}

def test_es_performance_benchmark():
    """Benchmark ES performance."""
    print("\n⚡ ES PERFORMANCE BENCHMARK")
    print("=" * 50)
    
    results = {}
    
    benchmark_configs = [
        ('identity_small', 'identity', {}, 16),
        ('identity_medium', 'identity', {}, 32),
        ('grid3_small', 'grid', {'grid_size': 3}, 16),
        ('grid5_small', 'grid', {'grid_size': 5}, 16),
        ('grid7_small', 'grid', {'grid_size': 7}, 16),
    ]
    
    for name, encoder_type, encoder_kwargs, hidden_size in benchmark_configs:
        try:
            print(f"\n📊 Benchmarking {name}...")
            
            from trainers.es_trainer import ModularESTrainer
            
            trainer = ModularESTrainer(
                encoder_type=encoder_type,
                hidden_size=hidden_size,
                **encoder_kwargs
            )
            
            rnn = trainer.rnn_template.from_params(trainer.rnn_template.get_params())
            
            # Benchmark evaluation speed
            eval_times = []
            for i in range(5):
                start_time = time.time()
                fitness = trainer.evaluate_rnn(rnn, seed=i, max_steps=50)
                eval_time = time.time() - start_time
                eval_times.append(eval_time)
            
            mean_time = np.mean(eval_times)
            
            print(f"   ⏱️  Avg evaluation time: {mean_time:.3f}s")
            print(f"   🧠 Parameters: {trainer.rnn_template.num_params}")
            print(f"   📊 Encoder output: {trainer.obs_dim}D")
            print(f"   🎯 Sample fitness: {fitness:.4f}")
            
            results[name] = {
                'eval_time': mean_time,
                'parameters': trainer.rnn_template.num_params,
                'encoder_dim': trainer.obs_dim,
                'sample_fitness': fitness
            }
            
        except Exception as e:
            print(f"   ❌ {name} benchmark failed: {e}")
            results[name] = {'error': str(e)}
    
    return results

def main():
    """Run comprehensive ES tests."""
    print("🚀 EVOLUTION STRATEGIES (ES) COMPREHENSIVE TEST")
    print("=" * 80)
    
    test_results = {}
    
    # Run all tests
    print("\n" + "="*80)
    test_results['components'] = test_es_components()
    
    print("\n" + "="*80)
    test_results['evaluation'] = test_es_evaluation()
    
    print("\n" + "="*80)
    test_results['mini_training'] = test_es_mini_training()
    
    print("\n" + "="*80)
    test_results['model_ops'] = test_es_model_operations()
    
    print("\n" + "="*80)
    test_results['benchmark'] = test_es_performance_benchmark()
    
    # Final summary
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE ES TEST RESULTS")
    print("="*80)
    
    # Component tests
    print("\n🔧 Component Tests:")
    for name, result in test_results['components'].items():
        status = "✅" if result['status'] == 'success' else "❌"
        print(f"  {status} {name}")
    
    # Evaluation tests
    print("\n🎯 Evaluation Tests:")
    for name, result in test_results['evaluation'].items():
        if result['status'] == 'success':
            print(f"  ✅ {name}: {result['fitness']:.4f} fitness, {result['eval_time']:.3f}s")
        else:
            print(f"  ❌ {name}: {result['error']}")
    
    # Training tests
    print("\n🏃 Mini Training Tests:")
    for name, result in test_results['mini_training'].items():
        if result['status'] == 'success':
            print(f"  ✅ {name}: {result['total_evaluations']} evals, {result['final_fitness']:.4f} fitness")
        else:
            print(f"  ❌ {name}: {result['error']}")
    
    # Model operations
    print("\n💾 Model Operations:")
    model_result = test_results['model_ops']
    if model_result['status'] == 'success':
        print(f"  ✅ Save/Load: {model_result['fitness_diff']:.6f} difference")
    else:
        print(f"  ❌ Save/Load: {model_result['error']}")
    
    # Performance benchmark
    print("\n⚡ Performance Benchmark:")
    for name, result in test_results['benchmark'].items():
        if 'error' not in result:
            print(f"  📊 {name}: {result['eval_time']:.3f}s/eval, {result['parameters']} params")
        else:
            print(f"  ❌ {name}: {result['error']}")
    
    # Overall assessment
    total_tests = sum(len(category) for category in test_results.values())
    successful_tests = sum(
        sum(1 for result in category.values() 
            if isinstance(result, dict) and result.get('status') == 'success')
        for category in [test_results['components'], test_results['evaluation'], test_results['mini_training']]
    )
    
    # Add model ops and benchmark successes
    if test_results['model_ops']['status'] == 'success':
        successful_tests += 1
    successful_tests += sum(1 for result in test_results['benchmark'].values() if 'error' not in result)
    
    print(f"\n🎉 FINAL RESULTS: {successful_tests}/{total_tests} tests passed")
    
    if successful_tests == total_tests:
        print("\n🌟 ALL ES SYSTEMS OPERATIONAL!")
        print("Your Evolution Strategies setup is fully functional!")
        print("\nNext steps:")
        print("1. Run full ES training:")
        print("   from trainers.es_trainer import ModularESTrainer")
        print("   trainer = ModularESTrainer(encoder_type='grid', grid_size=5)")
        print("   trainer.train(num_generations=1000)")
        print("2. Compare identity vs grid encoders")
        print("3. Experiment with different hidden sizes")
    else:
        failed_tests = total_tests - successful_tests
        print(f"\n⚠️  {failed_tests} tests failed. Check the details above.")
    
    return test_results

if __name__ == "__main__":
    test_results = main()