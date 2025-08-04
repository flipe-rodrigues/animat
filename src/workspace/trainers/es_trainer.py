import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import cma
import torch

from wrappers.es_wrapper import ESRNNAdapter
from envs.dm_env import make_arm_env
from encoders.encoders import ModalitySpecificEncoder, IdentityEncoder
from utils import encode_numpy, validate_compatibility

class ModularESTrainer:
    """Simplified ES trainer using DM-Control environment directly."""
    
    def __init__(self, encoder_type='identity', grid_size=5, hidden_size=25, alpha=0.1):
        
        # Set up encoder directly
        if encoder_type == 'grid':
            self.encoder = ModalitySpecificEncoder(grid_size=grid_size, raw_obs_dim=15)
        else:
            self.encoder = IdentityEncoder(obs_dim=15)
        
        self.obs_dim = self.encoder.output_dim
        self.action_dim = 4  # Confirmed: 4 muscle actuators
        
        # Create RNN adapter with correct action dimension
        self.rnn_template = ESRNNAdapter(
            input_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_size=hidden_size,
            alpha=alpha
        )
        
        # Validate compatibility
        validate_compatibility(self.encoder, self.rnn_template.policy)
        
        print(f"ES Trainer initialized:")
        print(f"  Encoder: {encoder_type} ({self.encoder.input_dim} → {self.encoder.output_dim})")
        print(f"  Action dimension: {self.action_dim} (deltoid, latissimus, biceps, triceps)")
        print(f"  RNN parameters: {self.rnn_template.num_params}")
        print(f"  Hidden size: {hidden_size}")
    
    def _extract_and_encode_obs(self, observation):
        """Extract DM-Control observation and encode it."""
        # Extract raw observation components
        muscle_data = observation['muscle_sensors']  # 12D
        target_pos = observation['target_position'].flatten()  # 3D (flattened from (1,3))
        raw_obs = np.concatenate([muscle_data, target_pos])
        
        # Encode using the encoder 
        encoded_obs = encode_numpy(self.encoder, raw_obs)
        return encoded_obs
    
    def evaluate_rnn(self, rnn, seed=0, max_steps=1000):
        """Evaluate single RNN episode using dm_env directly."""
        env = make_arm_env(random_seed=seed)
        
        rnn.init_state()
        total_reward = 0
        step_count = 0
        
        timestep = env.reset()
        
        while not timestep.last() and step_count < max_steps:
            # Extract and encode observation
            encoded_obs = self._extract_and_encode_obs(timestep.observation)
            
            # Get RNN action
            action = rnn.step(encoded_obs)
            
            # Ensure action is in valid range [0, 1] for muscle activations
            action = np.clip(action, 0.0, 1.0)
            
            # Step environment (dm_env handles all the reward logic)
            timestep = env.step(action)
            
            # Use dm_env's reward directly
            reward = timestep.reward if timestep.reward is not None else 0.0
            total_reward += reward
            step_count += 1
        
        env.close()
        return total_reward / step_count if step_count > 0 else 0
    
    def train(self, num_generations=1000, population_size=None, sigma=1.3, save_dir="./es_results"):
        """Train using CMA-ES."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize CMA-ES
        initial_params = self.rnn_template.get_params()
        es = cma.CMAEvolutionStrategy(initial_params, sigma)
        
        if population_size:
            es.opts['popsize'] = population_size
        
        fitnesses = []
        best_fitness = -np.inf
        best_params = None
        
        print(f"Starting ES training with population size: {es.opts['popsize']}")
        
        for generation in tqdm(range(num_generations), desc="ES Training"):
            solutions = es.ask()
            generation_fitnesses = []
            
            # Evaluate population
            for individual_idx, params in enumerate(solutions):
                rnn = self.rnn_template.from_params(params)
                
                # Evaluate fitness (average over multiple seeds)
                fitness_values = []
                for seed in range(3):
                    fitness = self.evaluate_rnn(rnn, seed=generation*100 + seed)
                    fitness_values.append(fitness)
                
                avg_fitness = np.mean(fitness_values)
                generation_fitnesses.append(avg_fitness)
                fitnesses.append((generation, individual_idx, avg_fitness))
                
                # Track best
                if avg_fitness > best_fitness:
                    best_fitness = avg_fitness
                    best_params = params.copy()
                
                if generation % 10 == 0:
                    print(f"Gen {generation}.{individual_idx}: {avg_fitness:.4f}")
            
            # Update CMA-ES (CMA-ES minimizes, so negate fitness)
            es.tell(solutions, [-f for f in generation_fitnesses])
            
            # Periodic saving
            if generation % 50 == 0 and best_params is not None:
                best_rnn = self.rnn_template.from_params(best_params)
                eval_fitness = self.evaluate_rnn(best_rnn, seed=0)
                print(f"Generation {generation}: Best fitness = {eval_fitness:.4f}")
                self.save_model(best_rnn, f"{save_dir}/best_rnn_gen_{generation}.pkl")
        
        # Final save
        if best_params is not None:
            best_rnn = self.rnn_template.from_params(best_params)
            self.save_model(best_rnn, f"{save_dir}/final_best_rnn.pkl")
        
        return best_params, fitnesses
    
    def save_model(self, rnn, filepath):
        """Save RNN model."""
        model_data = {
            'params': rnn.get_params(),
            'config': {
                'input_dim': rnn.policy.input_dim,
                'action_dim': rnn.policy.output_dim,
                'hidden_size': rnn.policy.hidden_size,
                'alpha': rnn.policy.alpha
            },
            'encoder_config': {
                'type': type(self.encoder).__name__,
                'input_dim': self.encoder.input_dim,
                'output_dim': self.encoder.output_dim
            }
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath):
        """Load RNN model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        rnn = ESRNNAdapter(**model_data['config'])
        rnn.set_params(model_data['params'])
        return rnn
    
    def evaluate_best(self, model_path, num_episodes=10, render=False):
        """Evaluate a saved model."""
        rnn = self.load_model(model_path)
        
        fitnesses = []
        for episode in range(num_episodes):
            fitness = self.evaluate_rnn(rnn, seed=episode)
            fitnesses.append(fitness)
            print(f"Episode {episode}: {fitness:.4f}")
        
        print(f"Average fitness over {num_episodes} episodes: {np.mean(fitnesses):.4f} ± {np.std(fitnesses):.4f}")
        return fitnesses