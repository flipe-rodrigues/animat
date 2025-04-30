import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from environment import make_arm_env
from shimmy_wrapper import DmControlCompatibilityV0, FlattenObservation, RescaleAction, SuccessInfoWrapper

VECNORM_PATH = "vec_normalize.pkl"
MODEL_PATH = "./best_model/best_model.zip"

def make_eval_env(seed=42):
    base = make_arm_env(random_seed=seed)
    env = DmControlCompatibilityV0(base)
    env = FlattenObservation(env)
    env = RescaleAction(env, -1.0, 1.0)
    env = SuccessInfoWrapper(env)
    env = Monitor(env)
    return env

def load_eval_env(vecnorm_path):
    if not os.path.exists(vecnorm_path):
        raise FileNotFoundError(f"VecNormalize file not found: {vecnorm_path}")
    eval_vec = DummyVecEnv([lambda: make_eval_env(42)])
    eval_vec = VecNormalize.load(vecnorm_path, eval_vec)
    eval_vec.training = False
    eval_vec.norm_reward = False
    return eval_vec

if __name__ == "__main__":
    # Load normalized evaluation env
    eval_vec = load_eval_env(VECNORM_PATH)

    # Load trained model
    model = SAC.load(MODEL_PATH)

    # Evaluate using SB3 utility
    mean_ret, std_ret = evaluate_policy(
        model, eval_vec, n_eval_episodes=20, deterministic=True
    )
    print(f"Mean reward over 20 eps: {mean_ret:.2f} ± {std_ret:.2f}")
