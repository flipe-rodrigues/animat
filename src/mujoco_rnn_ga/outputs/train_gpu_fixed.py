import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid threading issues
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from dm_testing.arm_env import load, _DEFAULT_TIME_LIMIT, _CONTROL_TIMESTEP
from dm_testing.dm_control_test import display_video
import os
import time
import matplotlib.animation as animation

# Add this import for CPU monitoring
try:
    import psutil
    has_psutil = True
except ImportError:
    has_psutil = False
    print("psutil not found - CPU monitoring will be disabled")

# Configure CUDA and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Test CUDA functionality
test_tensor = torch.zeros(1, device=device)
print(f"Test tensor device: {test_tensor.device}")
if torch.cuda.is_available():
    # Force some GPU computation
    test_model = torch.nn.Linear(100, 100).to(device)
    test_input = torch.randn(64, 100, device=device)
    test_output = test_model(test_input)
    print(f"GPU computation test successful: {test_output.shape}")
    print(f"GPU memory after test: {torch.cuda.memory_allocated()/1e6:.2f} MB")

# Enable mixed precision for better GPU performance
if torch.cuda.is_available():
    # For PyTorch 1.6 and later
    from torch.cuda.amp import GradScaler, autocast
    scaler = GradScaler('cuda')
    use_amp = True
    print("Mixed precision training enabled")
else:
    use_amp = False

# Advanced parallelism settings - OPTIMIZED for MuJoCo
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
os.environ["KMP_BLOCKTIME"] = "1"
# Since CPU is underutilized at 30%, use more cores for simulation
os.environ["OMP_NUM_THREADS"] = str(max(1, os.cpu_count() - 2))  # Use most cores
os.environ["MKL_NUM_THREADS"] = str(max(1, os.cpu_count() - 2))

# PyTorch threading settings
torch.set_num_threads(max(1, os.cpu_count() - 2))
torch.set_num_interop_threads(max(1, os.cpu_count() // 2))

# Enable optimizations
torch.backends.mkldnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere GPUs
torch.backends.cudnn.benchmark = True         # Optimize cudnn for fixed input sizes

class DMControlGymnasiumWrapper(gym.Env):
    """Convert dm_control environment to Gymnasium interface for Stable Baselines3."""
    
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode="rgb_array", verbose=False):
        """Initialize the wrapper."""
        # Switch to software rendering for WSL compatibility
        os.environ["MUJOCO_GL"] = "osmesa"  # Software rendering instead of EGL
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"  # Help OpenGL find the right backend
        os.environ["MPLBACKEND"] = "Agg"
        self.env = load()
        self.render_mode = render_mode
        self.verbose = verbose
        
        # Calculate max steps based on time limit and control timestep
        self.max_steps_per_episode = int(_DEFAULT_TIME_LIMIT / _CONTROL_TIMESTEP)
        if self.verbose:
            print(f"Max steps per episode: {self.max_steps_per_episode} " +
                  f"(time limit: {_DEFAULT_TIME_LIMIT}s, timestep: {_CONTROL_TIMESTEP}s)")
        
        # Define action space (muscle activations between 0 and 1)
        action_spec = self.env.action_spec()
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=action_spec.shape,
            dtype=np.float32
        )
        
        # Define observation space (flattened observations)
        total_size = 0
        for name, spec in self.env.observation_spec().items():
            if name == 'target_position':
                total_size += 3
            else:
                total_size += int(np.prod(spec.shape))
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_size,),
            dtype=np.float32
        )
        
        # For tracking
        self.steps = 0
        self.episode_reward = 0
        self.episode_count = 0
        self.target_changes = 0
    
    def reset(self, *, seed=None, options=None):
        """Reset the environment and randomize target."""
        if seed is not None:
            np.random.seed(seed)
        
        time_step = self.env.reset()
        self.steps = 0
        self.episode_reward = 0
        self.episode_count += 1
        
        # FORCE TARGET RANDOMIZATION
        # Get access to reachable positions
        task = self.env.task
        
        if hasattr(task, '_reachable_positions') and len(task._reachable_positions) > 0:
            # Explicitly select a random target position
            num_positions = len(task._reachable_positions)
            target_idx = np.random.randint(0, num_positions)
            new_target_position = task._reachable_positions[target_idx]
            
            # Get the old target position if possible
            try:
                # In DM Control, mocap_pos is a structured array
                old_target = self.env.physics.named.data.mocap_pos["target"].copy()
                
                if self.verbose:
                    print(f"Episode {self.episode_count}: Target change - " +
                          f"From {old_target} to {new_target_position}")
                    
                # Only increment counter if we successfully changed targets
                if not np.array_equal(old_target, new_target_position):
                    self.target_changes += 1
            except (KeyError, IndexError, ValueError, AttributeError):
                if self.verbose:
                    print(f"Episode {self.episode_count}: Setting initial target to {new_target_position}")
            
            # Apply the new target position to the MuJoCo mocap body
            self.env.physics.named.data.mocap_pos["target"] = new_target_position
            
            # Update internal task distance tracking
            if hasattr(task, 'prev_distance'):
                hand_position = self.env.physics.named.data.geom_xpos["hand"]
                task.prev_distance = np.linalg.norm(hand_position - new_target_position)
        elif self.verbose:
            print("WARNING: Could not access reachable_positions to randomize target")
        
        observation = self._process_observation(time_step.observation)
        return observation, {}
    
    def step(self, action):
        """Take a step in the environment."""
        # Just call step directly
        time_step = self.env.step(action)
        self.steps += 1

        # Force target to stay in 2D plane (Y=0)
        #target_pos = self.env.physics.named.data.mocap_pos['target']
        #target_pos[1] = 0  # Keep Y coordinate at 0
        #self.env.physics.named.data.mocap_pos['target'] = target_pos
        
        # Print debug info periodically
        if self.verbose and self.steps % 100 == 0:
            hand_position = self.env.physics.named.data.geom_xpos['hand']
            target_position = self.env.physics.named.data.mocap_pos['target']
            distance = np.linalg.norm(hand_position - target_position)
            print(f"Step {self.steps} (Ep {self.episode_count}):")
            print(f"  Hand: {hand_position}, Target: {target_position}")
            print(f"  Distance: {distance:.4f}, Y-plane: hand={hand_position[1]:.4f}, target={target_position[1]:.4f}")
            print(f"  Reward: {time_step.reward:.4f}")
        
        observation = self._process_observation(time_step.observation)
        reward = float(time_step.reward)
        self.episode_reward += reward
        
        # Check for episode termination:
        # 1. MuJoCo environment says episode is done (target reached or time limit)
        # 2. Explicit step count limit as backup
        terminated = time_step.last() 
        truncated = self.steps >= self.max_steps_per_episode
        
        info = {}
        if terminated or truncated:
            info['episode'] = {'r': self.episode_reward, 'l': self.steps, 'target_changes': self.target_changes}
            if self.verbose:
                print(f"Episode {self.episode_count} ended: " +
                      f"steps={self.steps}, reward={self.episode_reward:.2f}, " +
                      f"terminated={terminated}, truncated={truncated}")
        
        return observation, reward, terminated, truncated, info
    
    def _process_observation(self, observation):
        """Convert dm_control observation dict to flat array."""
        obs_list = []
        for name, obs in observation.items():
            if name == 'target_position':
                # Keep target position prominent in observations
                obs_list.append(obs.flatten()[:3].astype(np.float32))
            else:
                # Just flatten other observations
                obs_list.append(obs.flatten().astype(np.float32))
        
        # Simple concatenation - VecNormalize will handle normalization
        return np.concatenate(obs_list).astype(np.float32)
    
    def render(self):
        """CPU-only rendering substitute."""
        if self.render_mode == "rgb_array":
            # Skip actual rendering during training
            if not hasattr(self, '_is_evaluation') or not self._is_evaluation:
                return np.zeros((48, 64, 3), dtype=np.uint8)
            
            # Only use rendering during evaluation
            try:
                return self.env.physics.render(
                    camera_id=0, width=64, height=48, 
                    depth=False, segmentation=False
                )
            except Exception as e:
                print(f"Rendering error: {e}")
                return np.zeros((48, 64, 3), dtype=np.uint8)
        return None
    
    def close(self):
        """Close the environment."""
        self.env.close()

# Streamlined combined callback for efficiency
class EfficientTrackingCallback(BaseCallback):
    """Combined callback for tracking rewards and creating videos only at major milestones."""
    
    def __init__(self, eval_env, eval_freq=500000, verbose=1):  # Even less frequent (every 500k steps)
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq  # Render much less frequently (just 10 times in 5M steps)
        self.training_rewards = []
        self.timesteps = []
        self.episode_count = 0
        self.target_changes = 0
        self.best_mean_reward = -np.inf
        self.render_width = 320  # Lower resolution for faster rendering
        self.render_height = 240
    
    def _on_step(self):
        # Track episode completion with minimal overhead
        infos = self.locals.get('infos')
        dones = self.locals.get('dones')
        
        if infos is not None and dones is not None:
            for i, (done, info) in enumerate(zip(dones, infos)):
                if done and 'episode' in info:
                    # Just update counters with minimal processing
                    self.episode_count += 1
                    episode_reward = info['episode']['r']
                    self.training_rewards.append(episode_reward)
                    self.timesteps.append(self.num_timesteps)
                    
                    # Update target changes count
                    if 'target_changes' in info['episode']:
                        self.target_changes = max(self.target_changes, info['episode']['target_changes'])
                    
                    # Very minimal logging (only every 100 episodes)
                    if self.episode_count % 100 == 0:
                        recent_rewards = self.training_rewards[-100:] if len(self.training_rewards) >= 100 else self.training_rewards
                        avg_reward = sum(recent_rewards) / len(recent_rewards)
                        print(f"Episodes: {self.episode_count} | Avg reward: {avg_reward:.2f} | Steps: {self.num_timesteps}")
        
        # Only evaluate and render at major milestones
        if self.n_calls % self.eval_freq == 0:
            # Set a flag to enable actual rendering only during evaluation
            self.eval_env._is_evaluation = True
            
            print(f"\n--- EVALUATION AT {self.num_timesteps} STEPS ---")
            # Evaluate agent
            obs, _ = self.eval_env.reset()
            episode_reward = 0
            frames = []
            done = False
            steps = 0
            
            # Capture only 20 frames per evaluation to reduce overhead
            while not done and steps < 200:  # Limit max steps for evaluation
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated
                steps += 1
                
                # Capture frames extremely sparsely
                if steps % 20 == 0:  # Only 5% of frames for faster rendering
                    frames.append(self.eval_env.render())
            
            print(f"Evaluation reward: {episode_reward:.2f} after {steps} steps")
            
            # Save best model with minimal overhead
            if episode_reward > self.best_mean_reward:
                self.best_mean_reward = episode_reward
                print(f"New best model with reward {episode_reward:.2f}")
                self.model.save(f"best_model_gpu_{self.num_timesteps}")
            
            # Generate video only at major milestones
            if len(frames) > 5:
                filename = f"eval_{self.num_timesteps//1000}k.gif"
                display_video(frames, filename=filename, framerate=15)  # Lower framerate
            
            # Reset the flag after evaluation
            self.eval_env._is_evaluation = False
        
        # Every 50,000 steps, collect detailed profiling info
        if self.n_calls % 20000 == 0:
            print("\n--- PERFORMANCE PROFILE ---")
            
            # 1. Test environment step time
            start_time = time.time()
            for _ in range(100):
                act = self.eval_env.action_space.sample()
                self.eval_env.step(act)
            env_time = (time.time() - start_time) / 100
            print(f"Environment step time: {env_time*1000:.2f} ms")
            
            # 2. Test model prediction time
            obs = self.eval_env.observation_space.sample()
            start_time = time.time()
            for _ in range(100):
                with torch.no_grad():
                    self.model.predict(obs, deterministic=True)
            pred_time = (time.time() - start_time) / 100
            print(f"Model prediction time: {pred_time*1000:.2f} ms")
            
            # 3. Test rendering time
            start_time = time.time()
            for _ in range(10):
                self.eval_env.render()
            render_time = (time.time() - start_time) / 10
            print(f"Rendering time: {render_time*1000:.2f} ms")
            
            # 4. Report bottleneck analysis
            total = env_time + pred_time
            print(f"Environment: {env_time/total*100:.1f}% of step time")
            print(f"Prediction: {pred_time/total*100:.1f}% of step time")
            print(f"Rendering: {render_time/total*100:.1f}% of step time")
            
            # 5. GPU utilization when available
            if torch.cuda.is_available():
                print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
                print(f"GPU util: {torch.cuda.utilization(0)}%")
        
        # Add a safety check before trying to access self.parent.env
        if self.n_calls % 30 == 0 and hasattr(self, 'parent') and self.parent is not None and hasattr(self.parent, 'env'):
            # Process a batch of environment steps without heavy GPU work
            try:
                for _ in range(10):
                    self.parent.env.step([self.parent.env.action_space.sample() 
                                         for _ in range(self.parent.env.num_envs)])
            except Exception as e:
                # Silently fail rather than crashing the training
                if self.verbose > 0:
                    print(f"Warning: Could not run extra environment steps: {e}")
        
        return True

def make_env(rank, verbose=False):
    """Create an environment wrapper."""
    def _init():
        # Only make the first environment verbose for cleaner logs
        env = DMControlGymnasiumWrapper(verbose=verbose and rank == 0)
        return Monitor(env)
    return _init

def train_sac(total_timesteps=5000000):
    """Train a SAC agent with optimized settings for MuJoCo + GPU."""
    # Create raw environment for checking
    raw_env = DMControlGymnasiumWrapper(verbose=False)  # Reduce verbosity
    check_env(raw_env)
    
    # Try to modify MuJoCo rendering properties safely
    try:
        # Set MuJoCo physics parameters for maximum speed
        raw_env.env.physics.model.opt.solver = 1  # PGS solver (fastest)
        raw_env.env.physics.model.opt.iterations = 10  # Reduce iterations
        raw_env.env.physics.model.opt.tolerance = 1e-5  # More tolerant convergence
        
        # Attempt to set rendering quality to minimum
        if hasattr(raw_env.env.physics.model, 'vis'):
            if hasattr(raw_env.env.physics.model.vis, 'quality'):
                if hasattr(raw_env.env.physics.model.vis.quality, 'shadowsize'):
                    raw_env.env.physics.model.vis.quality.shadowsize = 32
                if hasattr(raw_env.env.physics.model.vis.quality, 'offsamples'):
                    raw_env.env.physics.model.vis.quality.offsamples = 1
                if hasattr(raw_env.env.physics.model.vis.quality, 'numslices'):
                    raw_env.env.physics.model.vis.quality.numslices = 8
        
        print("Applied MuJoCo physics and rendering optimizations")
    except Exception as e:
        print(f"Could not apply MuJoCo physics optimizations: {e}")
    
    # Dynamically set number of environments based on CPU count
    num_cpus = os.cpu_count()
    num_envs = max(2, min(num_cpus - 2, 8))  # More reasonable default
    print(f"Using {num_envs} environments based on {num_cpus} available CPU cores")
    
    # Use SubprocVecEnv instead of DummyVecEnv for true parallelism
    env = SubprocVecEnv([make_env(i, verbose=i==0) for i in range(num_envs)])
    
    # Apply VecNormalize
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )
    
    # Create evaluation environment (used sparingly)
    eval_env = DMControlGymnasiumWrapper(verbose=False)
    
    # Adjust batch size to be slightly smaller but process more frequently
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=1000000,
        batch_size=4096,  # Increase batch size
        learning_starts=25000,
        train_freq=(4, "step"),  # Train after EVERY step
        gradient_steps=8,  # Increase gradient steps
        ent_coef="auto",
        tensorboard_log="./sac_tensorboard/",
        policy_kwargs=dict(
            net_arch=dict(
                pi=[512, 512, 256],
                qf=[512, 512, 256]
            ),
            activation_fn=torch.nn.ReLU,
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs=dict(eps=1e-5),
        ),
        device=device
    )
    
    # Move replay buffer to GPU
    if torch.cuda.is_available():
        model.replay_buffer.to(device)
        print("Replay buffer moved to GPU")
    
    # Add after model creation:
    print("\n--- VERIFYING MODEL ON GPU ---")
    for name, module in model.policy.named_modules():
        if hasattr(module, 'weight') and hasattr(module.weight, 'device'):
            print(f"{name}: {module.weight.device}")

    # Force a single training step to verify gradients - with proper buffer check
    if hasattr(model.replay_buffer, 'size') and model.replay_buffer.size() >= model.batch_size:
        model._train_step()
        print("Performed test training step successfully")
    elif hasattr(model.replay_buffer, 'pos') and model.replay_buffer.pos > 0:
        # Alternative way to check if buffer has data
        print(f"Buffer has {model.replay_buffer.pos} samples, waiting for {model.batch_size}")
    else:
        print(f"Buffer is empty, will begin training after {model.learning_starts} steps")
    
    # Add timing metrics
    timing_stats = {
        "step_times": [],
        "train_times": [],
        "predict_times": [],
        "env_sync_times": [],
        "last_print": time.time(),
        "report_interval": 60,  # Print stats every 60 seconds
    }
    
    # Modify the learn call to include a custom callback for timing
    class TimingCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.step_start_time = None
            self.train_start_time = None
        
        def _on_step(self):
            if self.step_start_time is not None:
                step_time = time.time() - self.step_start_time
                timing_stats["step_times"].append(step_time)
                
                # Add this print statement to confirm training is happening
                print(f"Training step at {self.num_timesteps} steps")
                
                # Check if it's time to print stats
                current_time = time.time()
                if current_time - timing_stats["last_print"] > timing_stats["report_interval"]:
                    timing_stats["last_print"] = current_time
                    
                    # Calculate statistics
                    recent_steps = timing_stats["step_times"][-100:]
                    recent_trains = timing_stats["train_times"][-100:] if timing_stats["train_times"] else [0]
                    
                    avg_step = sum(recent_steps) / len(recent_steps)
                    avg_train = sum(recent_trains) / len(recent_trains) if recent_trains else 0
                    
                    # Get GPU memory usage
                    gpu_mem = f"{torch.cuda.memory_allocated()/1e9:.1f}/{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB" if torch.cuda.is_available() else "N/A"
                    
                    print(f"\nPERFORMANCE STATS at {self.num_timesteps} steps:")
                    print(f"Avg step time: {avg_step*1000:.2f}ms")
                    print(f"Avg train time: {avg_train*1000:.2f}ms")
                    print(f"Steps per second: {1/avg_step:.1f}")
                    print(f"Train/step ratio: {(avg_train/avg_step)*100:.1f}%")
                    print(f"GPU memory: {gpu_mem}")
                    
                    # Use psutil only if available
                    if has_psutil:
                        print(f"CPU utilization: {psutil.cpu_percent()}%")
                    
                    # Try to import nvidia-smi info if available
                    try:
                        import subprocess
                        result = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'])
                        gpu_util = result.decode('utf-8').strip()
                        print(f"GPU utilization: {gpu_util}%")
                    except:
                        pass
            
            # Start timing the next step
            self.step_start_time = time.time()
            return True
    
    # Create single streamlined callback
    efficient_callback = EfficientTrackingCallback(eval_env, eval_freq=500000)
    
    # Create checkpoint callback but with less frequency
    checkpoint_callback = CheckpointCallback(
        save_freq=500000,  # Save only every 500K steps
        save_path="./sac_checkpoints_gpu/",
        name_prefix="sac_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    # Add TimingCallback to your callbacks list
    callbacks = [efficient_callback, checkpoint_callback, TimingCallback()]
    
    # Print training config
    print("\n============ OPTIMIZED TRAINING CONFIGURATION ============")
    print(f"Time limit per episode: {_DEFAULT_TIME_LIMIT} seconds")
    print(f"Steps per episode: {int(_DEFAULT_TIME_LIMIT / _CONTROL_TIMESTEP)}")
    print(f"Environments: {num_envs} parallel environments")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Device: {device}")
    print(f"Batch size: {model.batch_size}")
    print(f"Training frequency: {model.train_freq}")
    print(f"Gradient steps: {model.gradient_steps}")
    print("========================================================\n")
    
    # Train
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
        tb_log_name="SAC_GPU_5M_Optimized",
    )
    total_time = time.time() - start_time
    
    # Print summary
    print(f"\nTraining completed in {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    print(f"Episodes completed: {efficient_callback.episode_count}")
    print(f"Target position changes: {efficient_callback.target_changes}")
    print(f"Best reward achieved: {efficient_callback.best_mean_reward:.2f}")
    print(f"Training throughput: {total_timesteps / total_time:.1f} steps/second")
    
    # Save final model
    model.save("final_sac_model_5M_gpu")
    env.save("vec_normalize_5M_gpu.pkl")
    
    # Final performance evaluation with high quality rendering
    print("Recording final performance video...")
    obs, _ = eval_env.reset()
    frames = []
    done = False
    step = 0
    total_reward = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        total_reward += reward
        done = terminated or truncated
        if step % 2 == 0:
            frames.append(eval_env.render())
        step += 1
    
    display_video(frames, filename="final_performance_gpu.gif", framerate=30)
    print(f"Final performance: {total_reward:.2f} reward")
    
    return model

if __name__ == "__main__":
    # Check if required packages are installed
    try:
        import stable_baselines3
        import gymnasium
    except ImportError:
        print("Installing required packages...")
        # For GPU support, install PyTorch with CUDA
        os.system("pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118")
        os.system("pip install gymnasium stable-baselines3[extra]")
        print("Packages installed. Please run the script again.")
        exit()
    
    # Print system info
    print(f"CPU Count: {os.cpu_count()} cores")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        # Check for GPU memory leaks during training
        torch.cuda.empty_cache()
    else:
        print("WARNING: CUDA not available! Training will use CPU only.")
    
    # Test the environment
    env = DMControlGymnasiumWrapper(verbose=True)
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")
    
    # Create output directories if they don't exist
    os.makedirs("./sac_checkpoints_gpu/", exist_ok=True)
    os.makedirs("./sac_tensorboard/", exist_ok=True)
    
    # Train the agent with 5M steps
    model = train_sac(total_timesteps=50000)
    print(f"Training completed. Time limit: {_DEFAULT_TIME_LIMIT} seconds")
    
    # Final memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cleared")
