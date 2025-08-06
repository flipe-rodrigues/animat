"""Complete system test for modular ES training."""

import torch
import numpy as np

def test_individual_components():
    """Test each component in isolation."""
    print("🔧 TESTING INDIVIDUAL COMPONENTS")
    print("=" * 50)
    
    # Test encoders
    print("\n1. Testing Encoders...")
    from encoders.encoders import IdentityEncoder, ModalitySpecificEncoder
    
    # Identity encoder
    identity = IdentityEncoder(obs_dim=15)
    test_obs = np.random.randn(15)
    encoded = identity(torch.FloatTensor(test_obs))
    print(f"✅ Identity: {test_obs.shape} → {encoded.shape}")
    
    # Grid encoder
    grid = ModalitySpecificEncoder(grid_size=5, raw_obs_dim=15)
    encoded_grid = grid(torch.FloatTensor(test_obs))
    print(f"✅ Grid: {test_obs.shape} → {encoded_grid.shape} (expected: 38)")
    
    # Test RNN
    print("\n2. Testing RNN...")
    from networks.rnn import RNNPolicy
    
    rnn = RNNPolicy(input_dim=15, action_dim=4, hidden_size=25)  # Correct: 4 actions
    obs_tensor = torch.FloatTensor(test_obs).unsqueeze(0)
    action, hidden = rnn(obs_tensor)
    print(f"✅ RNN: {obs_tensor.shape} → {action.shape}")
    
    # Test environment
    print("\n3. Testing Environment...")
    from envs.dm_env import make_arm_env
    
    try:
        env = make_arm_env(random_seed=42)
        timestep = env.reset()
        print(f"✅ Environment reset successful")
        print(f"   Observation keys: {list(timestep.observation.keys())}")
        print(f"   Muscle sensors shape: {timestep.observation['muscle_sensors'].shape}")
        print(f"   Target position shape: {timestep.observation['target_position'].shape}")
        
        # Test step with correct action size (4D muscle activations)
        action = np.random.uniform(0, 1, size=4)  # Valid muscle activations [0,1]
        timestep = env.step(action)
        print(f"✅ Environment step successful, reward: {timestep.reward:.4f}")
        env.close()
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_es_adapter():
    """Test the ES RNN adapter."""
    print("\n🔄 TESTING ES ADAPTER")
    print("=" * 50)
    
    try:
        from wrappers.es_wrapper import ESRNNAdapter
        
        # Create adapter with correct dimensions
        adapter = ESRNNAdapter(input_dim=15, action_dim=4, hidden_size=25)  # 4 actions
        print(f"✅ ES Adapter created: {adapter.num_params} parameters")
        
        # Test parameter flattening/unflattening
        params = adapter.get_params()
        print(f"✅ Parameter extraction: shape {params.shape}")
        
        # Test creating new instance
        new_adapter = adapter.from_params(params)
        print(f"✅ Parameter setting successful")
        
        # Test forward pass
        obs = np.random.randn(15)
        adapter.init_state()
        action = adapter.step(obs)
        print(f"✅ Forward pass: {obs.shape} → {action.shape}")
        print(f"   Action range: [{action.min():.3f}, {action.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ ES Adapter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test full integration."""
    print("\n🎯 TESTING FULL INTEGRATION")
    print("=" * 50)
    
    try:
        from trainers.es_trainer import ModularESTrainer
        
        # Test with identity encoder
        print("\n1. Identity Encoder Setup...")
        trainer_identity = ModularESTrainer(
            encoder_type='identity',
            hidden_size=16  # Small for speed
        )
        print("✅ Identity trainer created")
        
        # Test with grid encoder
        print("\n2. Grid Encoder Setup...")
        trainer_grid = ModularESTrainer(
            encoder_type='grid',
            grid_size=3,  # Small for speed
            hidden_size=16
        )
        print("✅ Grid trainer created")
        
        # Test single evaluation
        print("\n3. Single Evaluation Test...")
        rnn = trainer_identity.rnn_template.from_params(
            trainer_identity.rnn_template.get_params()
        )
        fitness = trainer_identity.evaluate_rnn(rnn, seed=0, max_steps=50)
        print(f"✅ Evaluation successful! Fitness: {fitness:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mini_training():
    """Test very short training run."""
    print("\n🏃 TESTING MINI TRAINING")
    print("=" * 50)
    
    try:
        from trainers.es_trainer import ModularESTrainer
        
        trainer = ModularESTrainer(
            encoder_type='identity',
            hidden_size=8  # Very small
        )
        
        print("Starting 2-generation training...")
        best_params, fitnesses = trainer.train(
            num_generations=2,
            sigma=0.5,
            save_dir="./test_es_results"
        )
        
        print(f"✅ Mini training completed! {len(fitnesses)} evaluations")
        return True
        
    except Exception as e:
        print(f"❌ Mini training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 COMPREHENSIVE SYSTEM TEST")
    print("=" * 80)
    
    # Run all tests
    tests_passed = 0
    total_tests = 4
    
    if test_individual_components():
        tests_passed += 1
        
    if test_es_adapter():
        tests_passed += 1
        
    if test_integration():
        tests_passed += 1
        
    if test_mini_training():
        tests_passed += 1
    
    print(f"\n{'=' * 80}")
    print(f"🎉 RESULTS: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✅ ALL SYSTEMS GO! Your modular ES training is ready!")
        print("\nNext steps:")
        print("1. Run full training: python -c 'from trainers.es_trainer import ModularESTrainer; t=ModularESTrainer(); t.train()'")
        print("2. Compare encoders: try encoder_type='grid' vs 'identity'")
    else:
        print("❌ Some tests failed. Check the output above.")