from gymnasium.envs.registration import register

register(
    id="SequentialReaching-v0",
    entry_point="custom_envs.sequential_reaching:SequentialReachingEnv",
)