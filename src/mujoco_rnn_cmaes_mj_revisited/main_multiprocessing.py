import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from plants import *
from encoders import *
from environments import *
from networks import *
from utils import *
from cmaes import CMA


# ------------------------------------------------------------
# Worker-side evaluation (MUST be top-level)
# ------------------------------------------------------------
def evaluate_candidate(x, gg):
    # --- create local env + model (process-safe) ---
    reacher = SequentialReacher(plant_xml_file="arm.xml")

    target_encoder = GridTargetEncoder(
        grid_size=8,
        x_bounds=reacher.get_workspace_bounds()[0],
        y_bounds=reacher.get_workspace_bounds()[1],
        sigma=0.25,
    )

    rnn = NeuroMuscularRNN(
        input_size_tgt=target_encoder.size,
        input_size_len=reacher.num_sensors_len,
        input_size_vel=reacher.num_sensors_vel,
        input_size_frc=reacher.num_sensors_frc,
        hidden_size=25,
        output_size=reacher.num_actuators,
        activation=tanh,
        smoothing_factor=alpha_from_tau(
            tau=10e-3,
            dt=reacher.model.opt.timestep,
        ),
    )

    env = SequentialReachingEnv(
        plant=reacher,
        target_encoder=target_encoder,
        target_duration_distro={"mean": 3, "min": 1, "max": 6},
        iti_distro={"mean": 1, "min": 0, "max": 3},
        num_targets=10,
        randomize_gravity=True,
        loss_weights={
            "distance": 1,
            "energy": 0.1,
            "ridge": 0,
            "lasso": 0,
        },
    )

    local_rnn = rnn.from_params(x)
    fitness = -env.evaluate(local_rnn, seed=gg)
    return x, fitness


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":

    # --- main-process objects (NOT shared with workers) ---
    reacher = SequentialReacher(plant_xml_file="arm.xml")
    target_encoder = GridTargetEncoder(
        grid_size=8,
        x_bounds=reacher.get_workspace_bounds()[0],
        y_bounds=reacher.get_workspace_bounds()[1],
        sigma=0.25,
    )

    rnn = NeuroMuscularRNN(
        input_size_tgt=target_encoder.size,
        input_size_len=reacher.num_sensors_len,
        input_size_vel=reacher.num_sensors_vel,
        input_size_frc=reacher.num_sensors_frc,
        hidden_size=25,
        output_size=reacher.num_actuators,
        activation=tanh,
        smoothing_factor=alpha_from_tau(
            tau=10e-3,
            dt=reacher.model.opt.timestep,
        ),
    )

    env = SequentialReachingEnv(
        plant=reacher,
        target_encoder=target_encoder,
        target_duration_distro={"mean": 3, "min": 1, "max": 6},
        iti_distro={"mean": 1, "min": 0, "max": 3},
        num_targets=10,
        randomize_gravity=True,
        loss_weights={
            "distance": 1,
            "energy": 0.1,
            "ridge": 0,
            "lasso": 0,
        },
    )

    optimizer = CMA(mean=rnn.get_params(), sigma=1.3)

    num_generations = 10000
    fitnesses = []
    num_workers = os.cpu_count()

    for gg in range(num_generations):
        solutions = []

        # --- sample population ---
        xs = [optimizer.ask() for _ in range(optimizer.population_size)]

        # --- parallel evaluation ---
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            results = pool.map(lambda x: evaluate_candidate(x, gg), xs)

            for ii, (x, fitness) in enumerate(results):
                solutions.append((x, fitness))
                fitnesses.append((gg, ii, fitness))
                print(f"#{gg}.{ii} {fitness}")

        # --- CMA update ---
        optimizer.tell(solutions)

        # --- diagnostics / rendering (main process only!) ---
        if gg % 10 == 0:
            best_rnn = rnn.from_params(optimizer.mean)
            env.evaluate(best_rnn, seed=0, render=True, log=True)
            env.plot()

        # --- checkpoint ---
        if gg % 1000 == 0:
            file = f"../../models/optimizer_gen_{gg}_cmaesv2.pkl"
            with open(file, "wb") as f:
                pickle.dump(optimizer, f)
