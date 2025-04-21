"""
Test suite for the arm reaching project.
"""
import os
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any

from environment import ArmReachingEnv
from rnn_model import SimpleRNN
from evolutionary_trainer import EvolutionaryTrainer
from target_generator import TargetGenerator

# Set a fixed seed for deterministic tests
@pytest.fixture
def rng_key():
    """Fixed random key for reproducible tests."""
    return jax.random.PRNGKey(42)

@pytest.fixture
def env():
    """Environment fixture."""
    # Check if fixed model exists, otherwise use the original
    model_path = "arm_fixed.xml" if os.path.exists("arm_fixed.xml") else "arm.xml"
    return ArmReachingEnv(model_path=model_path, episode_length=100, render=False)

@pytest.fixture
def rnn(env):
    """RNN fixture."""
    return SimpleRNN(
        input_size=env.input_dim,
        hidden_size=16,
        output_size=env.output_dim
    )

class TestRNNModel:
    """Tests for the RNN model."""
    
    def test_init_params(self, rnn, rng_key):
        """Test parameter initialization."""
        params = rnn.init_params(rng_key)
        assert "w_ih" in params
        assert "w_hh" in params
        assert "w_ho" in params
        assert "b_h" in params
        assert "b_o" in params
        
        # Check dimensions
        assert params["w_ih"].shape == (rnn.hidden_size, rnn.input_size)
        assert params["w_hh"].shape == (rnn.hidden_size, rnn.hidden_size)
        assert params["w_ho"].shape == (rnn.output_size, rnn.hidden_size)
        assert params["b_h"].shape == (rnn.hidden_size,)
        assert params["b_o"].shape == (rnn.output_size,)
    
    def test_predict(self, rnn, rng_key):
        """Test forward pass."""
        params = rnn.init_params(rng_key)
        
        # Create dummy input and hidden state
        inputs = jnp.ones((rnn.input_size,))
        h_prev = jnp.zeros((rnn.hidden_size,))
        
        # Run prediction
        outputs, h_next = rnn.predict(params, inputs, h_prev)
        
        # Check outputs
        assert outputs.shape == (rnn.output_size,)
        assert jnp.all(outputs >= 0) and jnp.all(outputs <= 1)  # sigmoid activation
        
        # Check hidden state
        assert h_next.shape == (rnn.hidden_size,)
        assert jnp.all(h_next >= -1) and jnp.all(h_next <= 1)  # tanh activation
    
    def test_flatten_unflatten(self, rnn, rng_key):
        """Test parameter flattening and unflattening."""
        original_params = rnn.init_params(rng_key)
        
        # Flatten parameters
        flat_params = rnn.flatten_params(original_params)
        
        # Unflatten parameters
        reconstructed_params = rnn.unflatten_params(flat_params)
        
        # Check that all parameters match
        for k in original_params:
            assert jnp.allclose(original_params[k], reconstructed_params[k])

class TestEnvironment:
    """Tests for the environment."""
    
    def test_reset(self, env, rng_key):
        """Test environment reset."""
        observation = env.reset(rng_key)
        
        # Check observation dimensions
        assert observation.shape == (env.input_dim,)
    
    def test_step(self, env, rng_key):
        """Test environment step."""
        env.reset(rng_key)
        
        # Create random actions
        action = jnp.array([0.5, 0.5, 0.5, 0.5])  # Middle of action range
        
        # Take a step
        observation, reward, done, info = env.step(action)
        
        # Check outputs
        assert observation.shape == (env.input_dim,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "distance" in info
        assert "end_effector_pos" in info
        assert "target_pos" in info
    
    def test_generate_random_target(self, env, rng_key):
        """Test random target generation."""
        target = env.generate_random_target(rng_key)
        
        # Check target dimensions
        assert target.shape == (3,)

class TestTargetGenerator:
    """Tests for the target generator."""
    
    def test_random_targets(self, rng_key):
        """Test random target generation."""
        generator = TargetGenerator(seed=42)
        targets = generator.generate_random_targets(n_targets=10)
        
        # Check output
        assert targets.shape == (10, 3)
    
    def test_grid_targets(self):
        """Test grid target generation."""
        generator = TargetGenerator(seed=42)
        targets = generator.generate_grid_targets(nx=2, ny=2, nz=2)
        
        # Check output
        assert targets.shape == (8, 3)  # 2x2x2 = 8 points
    
    def test_circular_targets(self):
        """Test circular target generation."""
        generator = TargetGenerator(seed=42)
        targets = generator.generate_circular_targets(n_targets=12)
        
        # Check output
        assert targets.shape == (12, 3)
        
        # Check that all points are on the same height
        reference_height = targets[0, 2]
        for i in range(1, 12):
            assert jnp.isclose(targets[i, 2], reference_height)
    
    def test_figure_eight(self):
        """Test figure eight pattern."""
        generator = TargetGenerator(seed=42)
        
        # Test individual point generation
        point_0 = generator.figure_eight(0.0)
        point_25 = generator.figure_eight(0.25)
        point_5 = generator.figure_eight(0.5)
        point_75 = generator.figure_eight(0.75)
        
        # Check dimensions
        assert point_0.shape == (3,)
        
        # Generate a complete figure eight
        figure_eight_fn = lambda t: generator.figure_eight(t, scale=0.5, plane='xz')
        targets = generator.generate_sequential_pattern(
            figure_eight_fn, n_cycles=1, points_per_cycle=20
        )
        
        # Check output
        assert targets.shape == (20, 3)

class TestTrainer:
    """Basic tests for the trainer functionality."""
    
    def test_initialization(self, env):
        """Test trainer initialization."""
        trainer = EvolutionaryTrainer(
            env=env,
            popsize=8,  # Small population for testing
            hidden_size=8,
            n_targets=2,
            steps_per_target=10,
            save_path="test_models",
            seed=42
        )
        
        # Check initialization
        assert trainer.popsize == 8
        assert trainer.hidden_size == 8
        assert trainer.n_targets == 2
        assert trainer.steps_per_target == 10
        
        # Clean up
        if os.path.exists("test_models"):
            import shutil
            shutil.rmtree("test_models")

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])