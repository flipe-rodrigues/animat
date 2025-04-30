import os
import numpy as np
import multiprocessing
import psutil
import time
import torch
import torch.nn as nn

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import SubprocVecEnv

from shimmy_wrapper import create_env, create_training_env, create_eval_env, set_seeds
from success_tracking import SimpleMetricsCallback
# Add these to the top of your train.py file
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

def print_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / (1024 * 1024):.1f} MB")

class StepCounterCallback(BaseCallback):
    def __init__(self, verbose=0, print_freq=100000):
        super().__init__(verbose)
        self.step_count = 0
        self.last_print = 0
        self.print_freq = print_freq
        self.start_time = None
        
    def _on_training_start(self):
        import time
        self.start_time = time.time()
    
    def _on_step(self):
        # Count steps across all environments
        self.step_count += self.training_env.num_envs
        
        # Print progress at regular intervals
        if self.step_count - self.last_print >= self.print_freq:
            import time
            elapsed = time.time() - self.start_time
            steps_per_sec = self.step_count / elapsed if elapsed > 0 else 0
            
            print(f"\n--- Training Progress ---")
            print(f"Total env steps: {self.step_count:,} / 3,000,000 ({self.step_count/3000000:.1%})")
            print(f"Steps/sec: {steps_per_sec:.1f}")
            print(f"Estimated time left: {(3000000-self.step_count)/steps_per_sec/60:.1f} minutes")
            # Add memory check
            print_memory_usage()
            print(f"------------------------\n")
            
            self.last_print = self.step_count
        return True

# Add this class right after your StepCounterCallback class
class ProfileCallback(BaseCallback):
    """Track training metrics to determine if GPU would help."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.timestep = 0
        self.last_time = None
        self.cumulative_time = 0
        
    def _on_training_start(self):
        self.last_time = time.time()
    
    def _on_step(self):
        # Every few updates, report timing information
        self.timestep += 1
        
        if self.timestep % 100 == 0:  # Adjust frequency as needed
            current_time = time.time()
            elapsed = current_time - self.last_time
            self.cumulative_time += elapsed
            
            # Track detailed timing information where possible
            if hasattr(self.model, "logger") and self.model.logger is not None:
                if hasattr(self.model.logger, "name_to_value"):
                    for key, value in self.model.logger.name_to_value.items():
                        if "loss" in key:
                            print(f"{key}: {value:.5f}")
                            
            print(f"\n--- Update {self.timestep} metrics ---")
            print(f"Time for last 100 updates: {elapsed:.2f}s")
            print(f"Average time per update: {elapsed/100:.4f}s")
            
            self.last_time = current_time
            
        return True

def benchmark_network_operations(model, venv, iterations=1000):
    """Benchmark just the network operations to see if GPU would help."""
    import time
    
    # Get a batch of observations
    obs = venv.reset()
    lstm_states = None
    episode_starts = np.ones((venv.num_envs,), dtype=bool)
    
    # Time forward passes (prediction)
    start_time = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            action, lstm_states = model.predict(
                obs, 
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=False
            )
    
    forward_time = time.time() - start_time
    print(f"Forward pass time: {forward_time:.4f}s ({forward_time/iterations*1000:.2f}ms per iteration)")
    
    # If >30% of time is spent in network operations, GPU would likely help significantly
    return forward_time / iterations

def create_vecnormalize_file(model_path, output_path="vec_normalize_rnn.pkl", num_steps=20000):
    """Create a VecNormalize statistics file by collecting environment data."""
    # Create environment
    num_cpu = max(1, int(0.75 * multiprocessing.cpu_count()))
    venv = create_training_env(num_envs=num_cpu)
    
    print(f"Collecting normalization statistics over {num_steps} steps...")
    
    # Reset environment
    obs = venv.reset()
    steps_done = 0
    
    # Collect steps with random actions to gather statistics
    while steps_done < num_steps:
        # Use random actions to collect diverse observations
        actions = np.random.uniform(-1, 1, size=(num_cpu, venv.action_space.shape[0]))
        obs, _, dones, _ = venv.step(actions)
        
        steps_done += num_cpu
        
        # Handle episode resets
        for i, done in enumerate(dones):
            if done:
                obs_i = venv.env_method('reset', indices=i)
                if obs_i is not None and len(obs_i) > 0:
                    single_obs = obs_i[0][0] if isinstance(obs_i[0], tuple) else obs_i[0]
                    obs[i, :] = single_obs
        
        if steps_done % 1000 == 0:
            print(f"Collected {steps_done}/{num_steps} steps")
    
    # Save VecNormalize statistics
    venv.save(output_path)
    print(f"VecNormalize statistics saved to {output_path}")
    return venv

def train_arm_rnn():
    """Train an agent using RecurrentPPO with LSTM policy."""
    # Use about 75% of available cores for environment processes
    num_cpu = max(1, int(0.75 * multiprocessing.cpu_count()))
    venv = create_training_env(num_envs=num_cpu)
    
    print(f"Number of CPU cores used: {num_cpu}")

    # Ensure full seed consistency
    base_seed = 42
    venv.seed(base_seed)
    
    # Access the actual VecNormalize instance
    vec_normalize = venv
    
    # Add normalization warm-up phase with random actions
    print("\n====== Starting normalization warm-up phase ======")
    obs = vec_normalize.reset()
    warm_up_steps = 10000

    for step_idx in range(warm_up_steps // num_cpu):
        actions = np.random.uniform(-1, 1, size=(num_cpu, venv.action_space.shape[0]))
        obs, rewards, dones, infos = vec_normalize.step(actions)
        # Reset any environments that are done
        for i, done in enumerate(dones):
            if done:
                obs_i = vec_normalize.env_method('reset', indices=i)
                if obs_i is not None and len(obs_i) > 0:
                    single_obs = obs_i[0][0] if isinstance(obs_i[0], tuple) else obs_i[0]
                    obs[i, :] = single_obs
                    
        if step_idx % 100 == 0:
            print(f"Warm-up progress: {step_idx * num_cpu}/{warm_up_steps} steps")

    print("====== Normalization warm-up complete ======\n")
    
    # Create RecurrentPPO agent with optimized configuration
    model = RecurrentPPO(
        "MlpLstmPolicy",
        venv,
        learning_rate=3e-4,
        n_steps=1024,          # Increased from 128 to 2048 for better learning
        batch_size=256,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,         # Encourage exploration
        verbose=1,
        tensorboard_log="./tensorboard_logs_rnn/",
        policy_kwargs=dict(
            net_arch=dict(     # Separate network architectures for policy and value
                pi=[256, 256], # Policy network
                vf=[256, 256]  # Value function network
            ),
            lstm_hidden_size=128,  # Increased from 64 to 128
            n_lstm_layers=1        # Number of LSTM layers
        )
    )
    
    # Benchmark network operations
    print("Benchmarking network operations...")
    avg_network_time = benchmark_network_operations(model, venv, iterations=100)
    print(f"Average network forward pass time: {avg_network_time*1000:.2f}ms")

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path="./models_rnn/",
        name_prefix="arm_rnn",
        save_replay_buffer=False,  # Don't save replay buffer (not used in PPO anyway)
        save_vecnormalize=False
    )
    
    # Create eval env with same normalization
    eval_env = create_eval_env(vec_normalize)

    # Use LSTM-aware evaluation callback
    eval_callback = RecurrentEvalCallback(
        eval_env,
        best_model_save_path="./best_model_rnn/",
        log_path="./eval_logs_rnn/",
        eval_freq=100000,
        deterministic=True,
        render=False
    )
    
    # Add SimpleMetricsCallback to the callback list
    success_metrics = SimpleMetricsCallback(verbose=1)

    # Add step counter with appropriate frequency for 16 envs
    # With 16 envs, you're collecting ~16K steps per update, so print every ~10 updates
    step_counter = StepCounterCallback(print_freq=30000)  

    # Replace the profile_callback function definition with:
    profile_cb = ProfileCallback()

    # Create your list of callbacks:
    callbacks = [checkpoint_callback, eval_callback, success_metrics, step_counter, profile_cb]
    
    # Train the agent
    model.learn(
        total_timesteps=3000000,
        callback=callbacks,
        log_interval=100000,
        progress_bar=True
    )
    
    # After training completes
    vec_normalize.save("vec_normalize_rnn.pkl")
    model.save("arm_rnn_final")
    return model

def continue_training_rnn(model_path="arm_rnn_final", vecnorm_path="vec_normalize_rnn.pkl", 
                          additional_steps=400000, new_lr=5e-5):  # Lower learning rate
    """Continue training with RecurrentPPO."""
    # Set seeds for reproducibility
    set_seeds(42)
    
    # Create environments
    num_cpu = max(1, int(0.75 * multiprocessing.cpu_count()))
    vec_env = SubprocVecEnv([
        lambda rank=i: create_env(random_seed=12345 + rank) 
        for i in range(num_cpu)
    ])
    
    # Load normalization stats
    vec_normalize = VecNormalize.load(vecnorm_path, vec_env)
    vec_normalize.training = True
    vec_normalize.norm_reward = False
    
    # Load the original model
    old_model = RecurrentPPO.load(model_path)
    
    # Create a NEW model with more focus on exploitation
    new_model = RecurrentPPO(
        "MlpLstmPolicy", 
        vec_normalize,
        learning_rate=new_lr,
        n_steps=1024,         # Keep same as original
        batch_size=256,
        n_epochs=10,          # More epochs per batch
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,       # Reduced clip range for more conservative updates
        ent_coef=0.005,       # Reduce entropy for more exploitation
        verbose=1,
        tensorboard_log="./tensorboard_logs_rnn_continued/",
        policy_kwargs=old_model.policy_kwargs
    )
    
    # Copy the weights from old to new model
    new_model.policy.load_state_dict(old_model.policy.state_dict())
    
    print(f"Fine-tuning RNN with learning rate: {new_lr}")
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path="./models_rnn_continued/",
        name_prefix="arm_rnn_continued",
        save_replay_buffer=False,  # Don't save replay buffer (not used in PPO anyway)
        save_vecnormalize=False
    )
    
    # Create evaluation environment
    eval_env = create_eval_env(vec_normalize)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model_rnn_continued/",
        log_path="./eval_logs_rnn_continued/",
        eval_freq=100000,
        deterministic=True,
        render=False
    )
    
    # Add metrics callback
    success_metrics = SimpleMetricsCallback(verbose=1)
    callbacks = [checkpoint_callback, eval_callback, success_metrics]
    
    # Continue training with new model
    new_model.learn(
        total_timesteps=additional_steps,
        callback=callbacks,
        log_interval=100000,
        progress_bar=True
    )
    
    # Save the model
    vec_normalize.save("vec_normalize_rnn_continued.pkl")
    new_model.save("arm_rnn_final_continued")
    
    return new_model

def evaluate_recurrent_policy(model, env, n_eval_episodes=10, deterministic=True, max_steps_per_episode=200):
    """Enhanced evaluation with better logging and stricter timeouts."""
    # Reset environment
    obs = env.reset()
    
    # Initialize LSTM states and episode tracking
    lstm_states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    
    # Track episodes
    episode_rewards = []
    episode_successes = []
    current_episode_rewards = np.zeros(env.num_envs)
    
    # Track active episodes and steps per episode
    active_episodes = np.ones(env.num_envs, dtype=bool)
    episode_count = 0
    total_steps = 0
    steps_in_episodes = np.zeros(env.num_envs, dtype=int)
    
    while episode_count < n_eval_episodes:
        # Get action using LSTM state management
        action, lstm_states = model.predict(
            obs, 
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=deterministic
        )
        
        # Step environment
        obs, rewards, dones, infos = env.step(action)
        total_steps += env.num_envs
        steps_in_episodes += active_episodes
        
        # Print evaluation progress
        if total_steps % 100 == 0:
            print(f"Eval progress: {episode_count}/{n_eval_episodes} episodes, {total_steps} steps")
        
        # Update rewards
        current_episode_rewards += rewards * active_episodes
        
        # Force termination for episodes that exceed the step limit
        for i in range(env.num_envs):
            if active_episodes[i] and steps_in_episodes[i] >= max_steps_per_episode:
                print(f"\nFORCE TERMINATION - Episode {episode_count}, steps: {steps_in_episodes[i]}")
                print(f"Current distance: {np.linalg.norm(env.get_attr('unwrapped')[i].task._get_hand_target_distance()):.4f}")
                dones[i] = True
        
        # Handle episode terminations
        for i, done in enumerate(dones):
            if done and active_episodes[i]:
                episode_rewards.append(current_episode_rewards[i])
                if "success" in infos[i]:
                    episode_successes.append(infos[i]["success"])
                else:
                    episode_successes.append(False)
                    
                current_episode_rewards[i] = 0
                steps_in_episodes[i] = 0
                active_episodes[i] = False
                episode_count += 1
                
                # If we've evaluated enough episodes, deactivate remaining envs
                if episode_count >= n_eval_episodes:
                    active_episodes = np.zeros(env.num_envs, dtype=bool)
        
        # Update episode starts for LSTM state resets
        episode_starts = dones
    
    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    success_rate = np.mean(episode_successes) if episode_successes else 0.0
    
    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Success rate: {success_rate:.2%}")
    
    return mean_reward, success_rate

# Custom evaluation callback that's aware of LSTM states
class RecurrentEvalCallback(EvalCallback):
    """Evaluation callback that properly handles LSTM states for recurrent policies"""
    
    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            print("\nRunning evaluation...", flush=True)
            start_time = time.time()
            
            try:
                # Get the evaluation results
                mean_reward, success_rate = evaluate_recurrent_policy(
                    self.model, 
                    self.eval_env, 
                    n_eval_episodes=self.n_eval_episodes,
                    deterministic=self.deterministic,
                    max_steps_per_episode=200  # Lower timeout
                )
                
                # For best model saving and callbacks
                self.last_mean_reward = mean_reward
                
                print(f"Evaluation completed in {time.time() - start_time:.1f}s", flush=True)
                
                # Only run parent implementation to handle best model saving
                if self.best_model_save_path is not None:
                    self._save_best_model_if_needed()
                    
                if self.callback is not None:
                    return self.callback._on_step()
                    
            except Exception as e:
                print(f"Error during evaluation: {e}")
                
            return True
        return True
    
if __name__ == "__main__":
    # Set seeds for reproducibility
    base_seed = set_seeds(42)
    
    # Validate the environment
    env = create_env()
    check_env(env)
    
    # Set to True to train from scratch or False to continue training
    from_scratch = False
    
    if from_scratch:
        # Train the RecurrentPPO agent from scratch
        model = train_arm_rnn()
    else:
        # Continue training from a saved RecurrentPPO model with a reduced learning rate
        model = continue_training_rnn(
            model_path="./models_rnn/arm_rnn_1600000_steps.zip",
            vecnorm_path="vec_normalize_rnn.pkl", 
            additional_steps=500000,
            new_lr=1.5e-4  # Set to half the original rate for fine-tuning
        )