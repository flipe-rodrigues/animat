import os
import numpy as np
import multiprocessing
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import SubprocVecEnv

from wrappers.sb3_wrapper import create_env, create_training_env, create_eval_env, set_seeds
from wrappers.success_tracking import SimpleMetricsCallback

def train_arm():
    """Train an agent using the wrapped environment."""
    # Use about 75% of available cores for environment processes
    num_cpu = max(1, int(0.75 * multiprocessing.cpu_count()))
    venv = create_training_env(num_envs=num_cpu)
    
    print(f"Number of CPU cores used: {num_cpu}")

    # Ensure full seed consistency for Gym/Monitor wrappers
    base_seed = 42
    venv.seed(base_seed)
    
    # Debug: Check what type we really have
    print(f"Training env type: {type(venv)}")
    print(f"Is VecNormalize? {isinstance(venv, VecNormalize)}")
    
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
                    # Always unpack the observation from (obs, info)
                    single_obs = obs_i[0][0] if isinstance(obs_i[0], tuple) else obs_i[0]
                    obs[i, :] = single_obs
                    
        if step_idx % 100 == 0:
            print(f"Warm-up progress: {step_idx * num_cpu}/{warm_up_steps} steps")

    print("====== Normalization warm-up complete ======\n")
    
    # Create SAC agent
    model = SAC(
        "MlpPolicy", 
        venv,
        learning_rate=3e-4,
        buffer_size=1000000,  # Keep 1M buffer
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=4,
        gradient_steps=4,
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256],
                qf=[256, 256]  
            ),
            use_expln=True,
        )
    )
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models2/",
        name_prefix="arm_sac"
    )
    
    # Create eval env with same normalization
    eval_env = create_eval_env(vec_normalize)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model2/",
        log_path="./eval_logs2/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    # Add SimpleMetricsCallback to the callback list
    success_metrics = SimpleMetricsCallback(verbose=1)
    callbacks = [checkpoint_callback, eval_callback, success_metrics]
    
    # Train the agent
    model.learn(
        total_timesteps=800000,
        callback=callbacks,
        log_interval=1000,
        progress_bar=True
    )
    
    # After training completes
    vec_normalize.save("vec_normalize2.pkl")
    model.save("arm_final2")
    return model

def continue_training(model_path="arm_final", vecnorm_path="vec_normalize.pkl", additional_steps=400000, new_lr=1.5e-4):
    """Continue training with a new SAC instance."""
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
    
    # Load the original model to get its weights
    old_model = SAC.load(model_path)
    
    # Create a NEW model with the desired learning rate
    new_model = SAC(
        "MlpPolicy", 
        vec_normalize,
        learning_rate=new_lr,  # Set new learning rate here
        buffer_size=1000000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=4,
        gradient_steps=4,
        verbose=1,
        tensorboard_log="./tensorboard_logs_continued/",
        policy_kwargs=old_model.policy_kwargs  # Keep same architecture
    )
    
    # Copy the weights from old to new model
    new_model.policy.load_state_dict(old_model.policy.state_dict())
    
    print(f"Fine-tuning with learning rate: {new_lr}")
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models_continued/",
        name_prefix="arm_sac_continued"
    )
    
    # Create evaluation environment
    eval_env = create_eval_env(vec_normalize)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model_continued/",
        log_path="./eval_logs_continued/",
        eval_freq=5000,
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
        log_interval=10000,
        progress_bar=True
    )
    
    # Save the model
    vec_normalize.save("vec_normalize_continued.pkl")
    new_model.save("arm_final_continued")
    
    return new_model

if __name__ == "__main__":
    # Set seeds for reproducibility
    base_seed = set_seeds(42)
    
    # Validate the environment
    env = create_env()
    check_env(env)
    
    # Set to False to continue training with randomized initialization
    from_scratch = True
    
    if from_scratch:
        # Train the agent from scratch
        model = train_arm()
    else:
        # Continue training from a saved model with a reduced learning rate
        model = continue_training(
            model_path="arm_final",
            vecnorm_path="vec_normalize.pkl", 
            additional_steps=400000,
            new_lr=1.5e-4  # Set to half the original rate for fine-tuning
        )