# %%
"""
.####.##.....##.########...#######..########..########..######.
..##..###...###.##.....##.##.....##.##.....##....##....##....##
..##..####.####.##.....##.##.....##.##.....##....##....##......
..##..##.###.##.########..##.....##.########.....##.....######.
..##..##.....##.##........##.....##.##...##......##..........##
..##..##.....##.##........##.....##.##....##.....##....##....##
.####.##.....##.##.........#######..##.....##....##.....######.
"""
import os
import copy
import time
import pickle
import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import beta

# %%
"""
.##.....##.########.####.##........######.
.##.....##....##.....##..##.......##....##
.##.....##....##.....##..##.......##......
.##.....##....##.....##..##........######.
.##.....##....##.....##..##.............##
.##.....##....##.....##..##.......##....##
..#######.....##....####.########..######.
"""


def exponential_kernel(tau, time):
    """Generates an exponential kernel parameterized by its mean"""
    lambda_ = 1 / tau
    kernel = lambda_ * np.exp(-lambda_ * time)
    return kernel / kernel.sum()


def truncated_exponential(mu, a, b, size=1):
    """Sample from a truncated exponential distribution using inverse CDF method."""
    lambda_ = 1 / mu
    U = np.random.uniform(0, 1, size)
    exp_a, exp_b = np.exp(-lambda_ * a), np.exp(-lambda_ * b)
    return np.array(-np.log((1 - U) * (exp_a - exp_b) + exp_b) / lambda_)


def sample_entropy(samples, base=2):
    """Compute entropy directly from a vector of samples."""
    _, counts = np.unique(samples, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log(probs) / np.log(base))


def beta_from_mean(mu, nu=5, num_samples=1):
    alpha = mu * nu
    beta_ = (1 - mu) * nu
    return beta.rvs(alpha, beta_, size=num_samples)


def logistic(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


def xavier_init(n_in, n_out):
    stddev = np.sqrt(1 / (n_in + n_out))
    return np.random.randn(n_out, n_in) * stddev


def he_init(n_in, n_out):
    """He (Kaiming) initialization for ReLU weights."""
    stddev = np.sqrt(2 / n_in)
    return np.random.randn(n_out, n_in) * stddev


def euclidean_distance(pos1, pos2):
    return np.sqrt(np.sum((pos1 - pos2) ** 2))


# %%
"""
.########..##....##.##....##
.##.....##.###...##.###...##
.##.....##.####..##.####..##
.########..##.##.##.##.##.##
.##...##...##..####.##..####
.##....##..##...###.##...###
.##.....##.##....##.##....##
"""


class RNN:
    def __init__(
        self, context_size, feedback_size, hidden_size, output_size, activation, alpha
    ):
        self.context_size = context_size
        self.feedback_size = feedback_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.phi = activation
        self.alpha = alpha
        if activation == relu:
            self.init_fcn = he_init
        else:
            self.init_fcn = xavier_init
        self.init_weights()
        self.init_biases()
        self.init_state()

    def init_weights(self):
        self.W_ctx = self.init_fcn(n_in=self.context_size, n_out=self.hidden_size)
        self.W_fbk = self.init_fcn(n_in=self.feedback_size, n_out=self.hidden_size)
        self.W_h = self.init_fcn(n_in=self.hidden_size, n_out=self.hidden_size)
        self.W_out = self.init_fcn(n_in=self.hidden_size, n_out=self.output_size)

    def init_biases(self):
        # self.b_ctx = np.zeros(self.context_size)
        # self.b_fbk = np.zeros(self.feedback_size)
        self.b_h = np.zeros(self.hidden_size)
        self.b_out = np.zeros(self.output_size)

    def init_state(self):
        """Reset hidden state between episodes"""
        self.ctx = np.zeros(self.context_size)
        self.fbk = np.zeros(self.feedback_size)
        self.h = np.zeros(self.hidden_size)
        self.out = np.zeros(self.output_size)

    def step(self, ctx, fbk):
        """Compute one RNN step"""
        h = (1 - self.alpha) * self.h + self.alpha * self.phi(
            self.W_ctx @ ctx + self.W_fbk @ fbk + self.W_h @ self.h + self.b_h
        )
        out = (1 - self.alpha) * self.out + self.alpha * logistic(
            self.W_out @ self.h + self.b_out
        )
        self.ctx = ctx
        self.fbk = fbk
        self.h = h
        self.out = out
        return out

    def step_old(self, ctx, fbk):
        """Compute one RNN step"""
        self.h = tanh(
            self.W_ctx @ ctx + self.W_fbk @ fbk + self.W_h @ self.h + self.b_h
        )
        output = logistic(self.W_out @ self.h + self.b_out)
        return output

    def flatten(self):
        return np.concatenate(
            [
                self.W_ctx.flatten(),
                self.W_fbk.flatten(),
                self.W_h.flatten(),
                self.W_out.flatten(),
                # self.b_ctx.flatten(),
                # self.b_fbk.flatten(),
                self.b_h.flatten(),
                self.b_out.flatten(),
            ]
        )

    def unflatten(self, flat_params):
        """Return a new RNN with weights and biases from flattened parameters."""
        idx = 0

        def extract(shape):
            nonlocal idx
            size = np.prod(shape)
            params = flat_params[idx : idx + size].reshape(shape)
            idx += size
            return params

        new_rnn = copy.deepcopy(self)
        new_rnn.W_ctx = extract((self.hidden_size, self.context_size))
        new_rnn.W_fbk = extract((self.hidden_size, self.feedback_size))
        new_rnn.W_h = extract((self.hidden_size, self.hidden_size))
        new_rnn.W_out = extract((self.output_size, self.hidden_size))
        # new_rnn.b_ctx = extract((self.context_size,))
        # new_rnn.b_fbk = extract((self.feedback_size,))
        new_rnn.b_h = extract((self.hidden_size,))
        new_rnn.b_out = extract((self.output_size,))
        return new_rnn


# %%
"""
.##.....##.##.....##.......##..#######...######...#######.
.###...###.##.....##.......##.##.....##.##....##.##.....##
.####.####.##.....##.......##.##.....##.##.......##.....##
.##.###.##.##.....##.......##.##.....##.##.......##.....##
.##.....##.##.....##.##....##.##.....##.##.......##.....##
.##.....##.##.....##.##....##.##.....##.##....##.##.....##
.##.....##..#######...######...#######...######...#######.
"""


class MuJoCoPlant:
    def __init__(self, model_path="../mujoco/arm_model.xml"):
        """Initialize Mujoco simulation"""
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.num_sensors = self.model.nsensor
        self.num_actuators = self.model.nu
        self.parse_targets()

    def parse_targets(self, targets_path="../src/targets.csv"):
        targets = pd.read_csv(targets_path).values
        H = np.histogram2d(targets[:, 0], targets[:, 2], bins=100)
        self.targets = []
        self.targets.append(np.argwhere(H[0] > 0))
        self.targets.append(H[1])
        self.targets.append(H[2])

    def sample_targets(self, num_samples=1):
        idcs2d = self.targets[0]
        x_edges = self.targets[1]
        y_edges = self.targets[2]
        sampled_idcs = idcs2d[
            np.random.choice(idcs2d.shape[0], num_samples, replace=False)
        ]
        sampled_x = x_edges[sampled_idcs[:, 0]]
        sampled_y = y_edges[sampled_idcs[:, 1]]
        positions = np.zeros((num_samples, 3))
        positions[:, 0] = sampled_x
        positions[:, 2] = sampled_y
        return positions

    def update_target(self, position):
        self.data.mocap_pos = position
        mujoco.mj_forward(self.model, self.data)

    def reset(self):
        """Reset limb state"""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def step(self, muscle_activations):
        """Apply torques and step simulation"""
        self.data.ctrl[:] = muscle_activations
        mujoco.mj_step(self.model, self.data)

    def get_obs(self):
        """Return joint angles, velocities, and end-effector position"""
        sensor_data = self.data.sensordata
        k = 100
        sensor_data[self.num_sensors // 3 * 2 :] /= k
        return sensor_data

    def get_pos(self, geom_name):
        """Return current position of the end effector"""
        geom_id = self.model.geom(geom_name).id
        return self.data.geom_xpos[geom_id][:].copy()


# %%
"""
..######..##.....##....###............########..######.
.##....##.###...###...##.##...........##.......##....##
.##.......####.####..##...##..........##.......##......
.##.......##.###.##.##.....##.#######.######....######.
.##.......##.....##.#########.........##.............##
.##....##.##.....##.##.....##.........##.......##....##
..######..##.....##.##.....##.........########..######.
"""


class CMAES:
    def __init__(
        self,
        num_parameters,
        lambda_,
        mean0,
        sigma0,
    ):
        self.num_parameters = num_parameters
        self.num_perturbations = lambda_

        self.mean = mean0
        self.sigma = sigma0

        self.C = np.eye(num_parameters)  # Covariance matrix
        self.pc = np.zeros(num_parameters)  # Evolution path
        self.ps = np.zeros(num_parameters)
        self.weights = np.log(lambda_ + 0.5) - np.log(np.arange(1, lambda_ + 1))
        self.weights /= self.weights.sum()
        self.mu_eff = 1 / np.sum(self.weights**2)  # Effective number of parents
        self.c1 = 2 / ((num_parameters + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(
            1 - self.c1,
            2
            * (self.mu_eff - 2 + 1 / self.mu_eff)
            / ((num_parameters + 2) ** 2 + self.mu_eff),
        )
        self.damps = (
            1
            + 2 * max(0, np.sqrt((self.mu_eff - 1) / (num_parameters + 1)) - 1)
            + self.c1
            + self.cmu
        )
        self.chiN = np.sqrt(num_parameters) * (
            1 - 1 / (4 * num_parameters) + 1 / (21 * num_parameters**2)
        )


# %%
"""
.########.##.....##..#######..##.......##.....##.########.####..#######..##....##
.##.......##.....##.##.....##.##.......##.....##....##.....##..##.....##.###...##
.##.......##.....##.##.....##.##.......##.....##....##.....##..##.....##.####..##
.######...##.....##.##.....##.##.......##.....##....##.....##..##.....##.##.##.##
.##........##...##..##.....##.##.......##.....##....##.....##..##.....##.##..####
.##.........##.##...##.....##.##.......##.....##....##.....##..##.....##.##...###
.########....###.....#######..########..#######.....##....####..#######..##....##
"""


class EvolveSequentialReacher:
    def __init__(
        self,
        target_duration,
        num_targets,
        num_individuals,
        num_generations,
        num_hidden_units,
        mutation_rate,
        learning_rate,
        activation,
        time_constant,
    ):
        self.target_duration = target_duration
        self.num_targets = num_targets
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.learning_rate = learning_rate
        self.env = MuJoCoPlant()

        self.rnn = RNN(
            context_size=3,
            feedback_size=self.env.num_sensors,
            hidden_size=num_hidden_units,
            output_size=self.env.num_actuators,
            activation=activation,
            alpha=self.env.model.opt.timestep / time_constant,
        )

        with open("../models/best_rnn_gen_curr.pkl", "rb") as f:
            self.rnn = pickle.load(f)

        self.parameters = self.rnn.flatten()
        self.num_parameters = len(self.parameters)
        self.num_perturbations = 4 + int(3 * np.log(self.num_parameters))

        self.cmaes = CMAES(
            num_parameters=self.num_parameters,
            lambda_=self.num_perturbations,
            mean0=self.parameters,
            sigma0=self.mutation_rate,
        )

    def evaluate(self, rnn, seed=0):
        """Evaluate fitness of a given RNN policy"""
        np.random.seed(seed)

        rnn.init_state()
        self.env.reset()
        target_positions = self.env.sample_targets(self.num_targets)
        target_durations = truncated_exponential(
            mu=self.target_duration[0],
            a=self.target_duration[1],
            b=self.target_duration[2],
            size=self.num_targets,
        )
        target_offset_times = target_durations.cumsum()
        trial_duration = target_durations.sum()
        total_reward = 0

        target_idx = 0
        self.env.update_target(target_positions[target_idx])

        while target_idx < self.num_targets:
            sensory_feedback = self.env.get_obs()
            muscle_activations = rnn.step(
                target_positions[target_idx], sensory_feedback
            )
            self.env.step(muscle_activations)
            distance = euclidean_distance(
                self.env.get_pos("hand"), target_positions[target_idx]
            )
            energy = muscle_activations.sum() * 0.1
            reward = -(distance + energy) * self.env.model.opt.timestep
            total_reward += reward

            if self.env.data.time > target_offset_times[target_idx]:
                target_idx += 1
                if seed < 100:
                    rnn.init_state()
                    self.env.reset()
                if target_idx < self.num_targets:
                    self.env.update_target(target_positions[target_idx])

        average_reward = total_reward / trial_duration
        return average_reward

    def render(self, rnn, seed=0):
        """Render a couple of trials for a set of RNN params"""
        np.random.seed(seed)

        with mujoco.viewer.launch_passive(self.env.model, self.env.data) as viewer:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
            viewer.cam.lookat[:] = [0, -1.5, -0.5]
            viewer.cam.azimuth = 90
            viewer.cam.elevation = 0
            viewer.sync()

            rnn.init_state()
            self.env.reset()
            target_positions = self.env.sample_targets(self.num_targets)
            target_durations = truncated_exponential(
                mu=self.target_duration[0],
                a=self.target_duration[1],
                b=self.target_duration[2],
                size=self.num_targets,
            )
            target_offset_times = target_durations.cumsum()

            target_idx = 0
            self.env.update_target(target_positions[target_idx])

            while viewer.is_running() and target_idx < self.num_targets:
                step_start = time.time()

                sensory_feedback = self.env.get_obs()
                muscle_activations = rnn.step(
                    target_positions[target_idx], sensory_feedback
                )
                self.env.step(muscle_activations)

                if self.env.data.time > target_offset_times[target_idx]:
                    target_idx += 1
                    if target_idx < self.num_targets:
                        self.env.update_target(target_positions[target_idx])

                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = self.env.model.opt.timestep - (
                    time.time() - step_start
                )
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

            viewer.close()

    def plot(self, rnn, seed=0):
        """Render a couple of trials for a set of RNN params"""
        np.random.seed(seed)

        time_data = []
        sensor_data = {
            "deltoid_length": [],
            "latissimus_length": [],
            "biceps_length": [],
            "triceps_length": [],
            "deltoid_velocity": [],
            "latissimus_velocity": [],
            "biceps_velocity": [],
            "triceps_velocity": [],
            "deltoid_force": [],
            "latissimus_force": [],
            "biceps_force": [],
            "triceps_force": [],
        }
        target_data = []
        distance_data = []
        energy_data = []
        reward_data = []
        fitness_data = []

        rnn.init_state()
        self.env.reset()
        target_positions = self.env.sample_targets(self.num_targets)
        target_durations = truncated_exponential(
            mu=self.target_duration[0],
            a=self.target_duration[1],
            b=self.target_duration[2],
            size=self.num_targets,
        )
        target_offset_times = target_durations.cumsum()
        trial_duration = target_durations.sum()
        total_reward = 0

        target_idx = 0
        self.env.update_target(target_positions[target_idx])

        while target_idx < self.num_targets:
            sensory_feedback = self.env.get_obs()
            muscle_activations = rnn.step(
                target_positions[target_idx], sensory_feedback
            )
            self.env.step(muscle_activations)
            distance = euclidean_distance(
                self.env.get_pos("hand"), target_positions[target_idx]
            )
            energy = np.mean(muscle_activations)
            reward = -distance * self.env.model.opt.timestep
            total_reward += reward

            time_data.append(self.env.data.time)
            distance_data.append(distance)
            energy_data.append(energy)
            reward_data.append(reward)
            fitness_data.append(total_reward)
            target_data.append(target_positions[target_idx])
            for i, key in enumerate(sensor_data.keys()):
                sensor_data[key].append(sensory_feedback[i])

            if self.env.data.time > target_offset_times[target_idx]:
                target_idx += 1
                if target_idx < self.num_targets:
                    self.env.update_target(target_positions[target_idx])

        # Plot data
        fig, axes = plt.subplots(2, 2)

        # Length
        axes[0, 0].plot(time_data, sensor_data["deltoid_length"], label="Deltoid")
        axes[0, 0].plot(time_data, sensor_data["latissimus_length"], label="Latissimus")
        axes[0, 0].plot(time_data, sensor_data["biceps_length"], label="Biceps")
        axes[0, 0].plot(time_data, sensor_data["triceps_length"], label="Triceps")
        axes[0, 0].set_title("Length")

        # Velocity
        axes[0, 1].plot(time_data, sensor_data["deltoid_velocity"], label="Deltoid")
        axes[0, 1].plot(
            time_data, sensor_data["latissimus_velocity"], label="Latissimus"
        )
        axes[0, 1].plot(time_data, sensor_data["biceps_velocity"], label="Biceps")
        axes[0, 1].plot(time_data, sensor_data["triceps_velocity"], label="Triceps")
        axes[0, 1].set_title("Velocity")
        axes[0, 1].legend(loc="center left", bbox_to_anchor=(1, 0.5))

        # Force
        axes[1, 0].plot(time_data, sensor_data["deltoid_force"], label="Deltoid")
        axes[1, 0].plot(time_data, sensor_data["latissimus_force"], label="Latissimus")
        axes[1, 0].plot(time_data, sensor_data["biceps_force"], label="Biceps")
        axes[1, 0].plot(time_data, sensor_data["triceps_force"], label="Triceps")
        axes[1, 0].set_title("Force")

        # Fitness
        axes[1, 1].plot(time_data, distance_data, label="Distance")
        axes[1, 1].plot(time_data, reward_data, label="Reward")
        axes[1, 1].plot(time_data, energy_data, label="Energy")
        axes[1, 1].set_title("Fitness")
        axes[1, 1].set_ylim([-1.05, 1.05])
        axes[1, 1].legend(loc="lower left")

        # Create a twin axis (right y-axis)
        ax_right = axes[1, 1].twinx()
        ax_right.plot(
            time_data,
            fitness_data / trial_duration,
            color="r",
        )
        ax_right.set_ylabel("Cumulative Reward", color="r")
        ax_right.tick_params(axis="y", labelcolor="r")

        # Set axis labels
        for ax in axes.flat:
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Arb.")

        plt.tight_layout()
        plt.show()

    def evolve(self):
        """Run evolutionary learning process"""
        for gg in range(self.num_generations):

            # Generate new population
            Z = np.random.randn(self.cmaes.num_perturbations, self.cmaes.num_parameters)
            X = (
                self.cmaes.mean
                + self.cmaes.sigma * Z @ np.linalg.cholesky(self.cmaes.C).T
            )
            fitness = np.array(
                [
                    self.evaluate(self.rnn.unflatten(x), seed=gg)
                    for x in tqdm(X, desc="Evaluating")
                ]
            )

            # Sort by fitness and update mean
            sorted_idcs = np.argsort(fitness)[::-1]
            X = X[sorted_idcs]
            Z = Z[sorted_idcs]
            self.cmaes.mean = np.sum(
                self.cmaes.weights[:, np.newaxis] * X[: self.cmaes.num_perturbations],
                axis=0,
            )

            # Update evolution paths
            z_mean = np.sum(
                self.cmaes.weights[:, np.newaxis] * Z[: self.cmaes.num_perturbations],
                axis=0,
            )
            self.cmaes.ps = (1 - 0.5) * self.cmaes.ps + np.sqrt(
                0.5 * (2 - 0.5)
            ) * z_mean
            hsig = (
                np.linalg.norm(self.cmaes.ps) / np.sqrt(1 - (1 - 0.5) ** (2 * (gg + 1)))
                < (1.4 + 2 / (self.cmaes.num_parameters + 1)) * self.cmaes.chiN
            )
            self.cmaes.pc = (1 - self.cmaes.c1) * self.cmaes.pc + hsig * np.sqrt(
                self.cmaes.c1 * (2 - self.cmaes.c1)
            ) * z_mean

            # Update covariance matrix
            C_new = (
                1 - self.cmaes.c1 - self.cmaes.cmu
            ) * self.cmaes.C + self.cmaes.c1 * np.outer(self.cmaes.pc, self.cmaes.pc)
            for i in range(self.cmaes.num_perturbations):
                C_new += self.cmaes.cmu * self.cmaes.weights[i] * np.outer(Z[i], Z[i])
            self.cmaes.C = C_new

            # Update step size
            self.cmaes.sigma *= np.exp(
                (np.linalg.norm(self.cmaes.ps) / self.cmaes.chiN - 1)
                * 0.5
                / self.cmaes.damps
            )

            # Print best fitness in each generation
            print(f"Generation {gg+1}: Best fitness = {fitness.max()}")

            # Save
            if gg % 10 == 0:
                self.plot(self.rnn.unflatten(self.cmaes.mean))
            if gg % 50 == 0:
                self.render(self.rnn.unflatten(self.cmaes.mean))
            if gg % 100 == 0:
                file = f"../models/best_rnn_gen_{gg}_CMA-ES.pkl"
            else:
                file = f"../models/best_rnn_gen_curr_CMA-ES.pkl"
            with open(file, "wb") as f:
                pickle.dump(self.rnn, f)

        return self.mean  # Return best found solution


# %%
"""
.##.....##....###....####.##....##
.###...###...##.##....##..###...##
.####.####..##...##...##..####..##
.##.###.##.##.....##..##..##.##.##
.##.....##.#########..##..##..####
.##.....##.##.....##..##..##...###
.##.....##.##.....##.####.##....##
"""
# GPU parallel..
# time constants..
# activation functions??
# plot input space..
# target/trial durations..
# ES??
if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    reacher = EvolveSequentialReacher(
        target_duration=(4, 2, 6),
        num_targets=15,
        num_individuals=100,
        num_generations=1000,
        num_hidden_units=10,
        mutation_rate=0.01,
        learning_rate=0.0005,
        activation=tanh,
        time_constant=10e-3,
    )
    best_rnn = reacher.evolve()

# %%
"""
.########..########.##....##.########..########.########.
.##.....##.##.......###...##.##.....##.##.......##.....##
.##.....##.##.......####..##.##.....##.##.......##.....##
.########..######...##.##.##.##.....##.######...########.
.##...##...##.......##..####.##.....##.##.......##...##..
.##....##..##.......##...###.##.....##.##.......##....##.
.##.....##.########.##....##.########..########.##.....##
"""
models_dir = "../models"
gen_idx = 800  # Specify the generation index you want to plot
model_file = f"best_rnn_gen_{gen_idx}_CMA-ES.pkl"
# model_file = "best_rnn_gen_curr_CMA-ES.pkl"
with open(os.path.join(models_dir, model_file), "rb") as f:
    best_rnn = pickle.load(f)
reacher.render(best_rnn)
reacher.plot(best_rnn)