from evojax.algo import PGPE

def get_optimizer(policy, pop_size=64, init_stdev=0.1, seed=0):
    """Create a PGPE evolutionary optimization algorithm for the policy.
    
    PGPE (Policy Gradients with Parameter-based Exploration) is an effective
    evolutionary strategy for training neural networks.
    
    Args:
        policy: The policy to optimize.
        pop_size: Population size.
        init_stdev: Initial standard deviation.
        seed: Random seed.
        
    Returns:
        PGPE evolutionary optimization algorithm.
    """
    return PGPE(
        pop_size=pop_size,
        param_size=policy.params_size,
        init_params=None,
        optimizer=None,
        optimizer_config=None,
        center_learning_rate=0.15,
        stdev_learning_rate=0.1,
        init_stdev=init_stdev,
        stdev_max_change=0.2,
        solution_ranking=True,
        seed=seed
    )