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


def get_root_path():
    root_path = os.path.abspath(os.path.dirname(__file__))
    while root_path != os.path.dirname(root_path):
        if os.path.exists(os.path.join(root_path, ".git")):
            break
        root_path = os.path.dirname(root_path)
    return root_path


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


def manhattan_distance(pos1, pos2):
    return np.sum(np.abs(pos1 - pos2))


def normalize01(x, xmin, xmax, default=0.5):
    valid = xmax > xmin
    xnorm = np.full_like(x, default)
    xnorm[valid] = (x[valid] - xmin[valid]) / (xmax[valid] - xmin[valid])
    return xnorm


def zscore(x, xmean, xstd, default=0):
    valid = xstd > 0
    xnorm = np.full_like(x, default)
    xnorm[valid] = (x[valid] - xmean[valid]) / xstd[valid]
    return xnorm


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
        self.b_ctx = np.zeros(self.context_size)
        self.b_fbk = np.zeros(self.feedback_size)
        self.b_h = np.zeros(self.hidden_size)
        self.b_out = np.zeros(self.output_size)

    def init_state(self):
        """Reset hidden state between episodes"""
        self.ctx = np.zeros(self.context_size)
        self.fbk = np.zeros(self.feedback_size)
        self.h = np.zeros(self.hidden_size)
        self.out = np.zeros(self.output_size)

    def step_tau(self, ctx, fbk):
        """Compute one RNN step"""
        h = (1 - self.alpha) * self.h + self.alpha * self.phi(
            self.W_ctx @ ctx + self.W_fbk @ fbk + self.W_h @ self.h + self.b_h
        )
        out = (1 - self.alpha) * self.out + self.alpha * logistic(
            self.W_out @ self.h + self.b_out
        )
        self.h = h
        self.out = out
        return out

    def step(self, ctx, fbk):
        """Compute one RNN step"""
        self.h = tanh(
            self.W_ctx @ ctx + self.W_fbk @ fbk + self.W_h @ self.h + self.b_h
        )
        output = logistic(self.W_out @ self.h + self.b_out)
        return output

    @staticmethod
    def recombine(p1, p2):
        child = RNN(
            p1.context_size,
            p1.feedback_size,
            p1.hidden_size,
            p1.output_size,
            p1.phi,
            p1.alpha,
        )
        child.W_ctx = RNN.recombine_matrices(p1.W_ctx, p2.W_ctx)
        child.W_fbk = RNN.recombine_matrices(p1.W_fbk, p2.W_fbk)
        child.W_h = RNN.recombine_matrices(p1.W_h, p2.W_h)
        child.W_out = RNN.recombine_matrices(p1.W_out, p2.W_out)
        child.b_ctx = RNN.recombine_matrices(p1.b_ctx, p2.b_ctx)
        child.b_fbk = RNN.recombine_matrices(p1.b_fbk, p2.b_fbk)
        child.b_h = RNN.recombine_matrices(p1.b_h, p2.b_h)
        child.b_out = RNN.recombine_matrices(p1.b_out, p2.b_out)
        return child

    @staticmethod
    def recombine_matrices(A, B):
        mask = np.random.rand(*A.shape) > 0.5
        return np.where(mask, A, B)

    def mutate(self, rate):
        mutant = copy.deepcopy(self)
        mutant.W_ctx += self.init_fcn(mutant.context_size, mutant.hidden_size) * rate
        mutant.W_fbk += self.init_fcn(mutant.feedback_size, mutant.hidden_size) * rate
        mutant.W_h += self.init_fcn(mutant.hidden_size, mutant.hidden_size) * rate
        mutant.W_out += self.init_fcn(mutant.hidden_size, mutant.output_size) * rate
        mutant.b_ctx += np.random.randn(mutant.context_size) * rate
        mutant.b_fbk += np.random.randn(mutant.feedback_size) * rate
        mutant.b_h += np.random.randn(mutant.hidden_size) * rate
        mutant.b_out += np.random.randn(mutant.output_size) * rate
        return mutant


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
    def __init__(self, xml_file="arm_model.xml"):
        """Initialize Mujoco simulation"""

        mj_dir = os.path.join(get_root_path(), "mujoco")

        xml_path = os.path.join(mj_dir, xml_file)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.num_sensors = self.model.nsensor
        self.num_actuators = self.model.nu
        self.viewer = None

        # Get the site ID using the name of your end effector
        self.hand_id = self.model.geom("hand").id

        # Load sensor stats
        sensor_stats_path = os.path.join(mj_dir, "sensor_stats.pkl")
        with open(sensor_stats_path, "rb") as f:
            self.sensor_stats = pickle.load(f)
        print(self.sensor_stats)

        # Load target stats
        target_stats_path = os.path.join(mj_dir, "target_stats.pkl")
        with open(target_stats_path, "rb") as f:
            self.target_stats = pickle.load(f)
        print(self.target_stats)

        # Load valid target positions
        reachable_positions_path = os.path.join(mj_dir, "reachable_positions.pkl")
        with open(reachable_positions_path, "rb") as f:
            self.reachable_positions = pickle.load(f)

    def sample_targets(self, num_samples=10):
        return self.reachable_positions.sample(num_samples).values

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
        target_position = self.data.mocap_pos[0].copy()
        sensor_data = self.data.sensordata.copy()
        norm_target_position = zscore(
            target_position,
            self.target_stats["mean"].values,
            self.target_stats["std"].values,
        )
        norm_sensor_data = zscore(
            sensor_data,
            self.sensor_stats["mean"].values,
            self.sensor_stats["std"].values,
        )
        return norm_target_position, norm_sensor_data

    def get_hand_pos(self):
        return self.data.geom_xpos[self.hand_id][:].copy()

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
            self.viewer.cam.lookat[:] = [0, -1.5, -0.5]
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = 0
        else:
            if self.viewer.is_running():
                self.viewer.sync()
                time.sleep(self.model.opt.timestep)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


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
        activation,
        tau,
    ):
        self.target_duration = target_duration
        self.num_targets = num_targets
        self.num_individuals = num_individuals
        self.num_generations = num_generations
        self.num_hidden_units = num_hidden_units
        self.mutation_rate = mutation_rate
        self.env = MuJoCoPlant()
        self.population = [
            RNN(
                context_size=3,
                feedback_size=self.env.num_sensors,
                hidden_size=self.num_hidden_units,
                output_size=self.env.num_actuators,
                activation=activation,
                alpha=self.env.model.opt.timestep / tau,
            )
            for _ in range(num_individuals)
        ]
        self.logger = None

    def log(
        self,
        time,
        sensors,
        target,
        distance,
        energy,
        reward,
        fitness,
    ):
        if self.logger is None:
            self.logger = {}
            self.logger["time"] = []
            self.logger["sensors"] = {
                "deltoid_len": [],
                "latissimus_len": [],
                "biceps_len": [],
                "triceps_len": [],
                "deltoid_vel": [],
                "latissimus_vel": [],
                "biceps_vel": [],
                "triceps_vel": [],
                "deltoid_frc": [],
                "latissimus_frc": [],
                "biceps_frc": [],
                "triceps_frc": [],
            }
            self.logger["target"] = []
            self.logger["distance"] = []
            self.logger["energy"] = []
            self.logger["reward"] = []
            self.logger["fitness"] = []

        self.logger["time"].append(time)
        for i, key in enumerate(self.logger["sensors"].keys()):
            self.logger["sensors"][key].append(sensors[i])
        self.logger["target"].append(target)
        self.logger["distance"].append(distance)
        self.logger["energy"].append(energy)
        self.logger["reward"].append(reward)
        self.logger["fitness"].append(fitness)

    def evaluate(self, rnn, seed=0, render=False, log=False):
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
            if render:
                self.env.render()

            context, feedback = self.env.get_obs()
            muscle_activations = rnn.step(context, feedback)
            self.env.step(muscle_activations)
            hand_position = self.env.get_hand_pos()
            target_position = target_positions[target_idx]
            distance = manhattan_distance(hand_position, target_position)
            reward = -distance
            total_reward += reward

            if log:
                self.log(
                    time=self.env.data.time,
                    sensors=feedback,
                    target=target_position,
                    distance=distance,
                    energy=np.mean(muscle_activations),
                    reward=reward,
                    fitness=total_reward / trial_duration,
                )

            if self.env.data.time > target_offset_times[target_idx]:
                target_idx += 1
                if target_idx < self.num_targets:
                    self.env.update_target(target_positions[target_idx])

        self.env.close()

        return total_reward / trial_duration

    def plot(self):
        log = self.logger

        _, axes = plt.subplots(2, 2)

        # Length
        axes[0, 0].plot(log["time"], log["sensors"]["deltoid_len"], label="Deltoid")
        axes[0, 0].plot(
            log["time"],
            log["sensors"]["latissimus_len"],
            label="Latissimus",
        )
        axes[0, 0].plot(log["time"], log["sensors"]["biceps_len"], label="Biceps")
        axes[0, 0].plot(log["time"], log["sensors"]["triceps_len"], label="Triceps")
        axes[0, 0].set_title("Length")

        # Velocity
        axes[0, 1].plot(log["time"], log["sensors"]["deltoid_vel"], label="Deltoid")
        axes[0, 1].plot(
            log["time"],
            log["sensors"]["latissimus_vel"],
            label="Latissimus",
        )
        axes[0, 1].plot(log["time"], log["sensors"]["biceps_vel"], label="Biceps")
        axes[0, 1].plot(log["time"], log["sensors"]["triceps_vel"], label="Triceps")
        axes[0, 1].set_title("Velocity")
        axes[0, 1].legend(loc="center left", bbox_to_anchor=(1, 0.5))

        # Force
        axes[1, 0].plot(log["time"], log["sensors"]["deltoid_frc"], label="Deltoid")
        axes[1, 0].plot(
            log["time"], log["sensors"]["latissimus_frc"], label="Latissimus"
        )
        axes[1, 0].plot(log["time"], log["sensors"]["biceps_frc"], label="Biceps")
        axes[1, 0].plot(log["time"], log["sensors"]["triceps_frc"], label="Triceps")
        axes[1, 0].set_title("Force")

        # Fitness
        axes[1, 1].plot(log["time"], log["distance"], label="Distance")
        axes[1, 1].plot(log["time"], log["reward"], label="Reward")
        axes[1, 1].plot(log["time"], log["energy"], label="Energy")
        axes[1, 1].set_title("Fitness")
        axes[1, 1].set_ylim([-1.05, 1.05])
        axes[1, 1].legend(loc="lower left")

        # Create a twin axis (right y-axis)
        ax_right = axes[1, 1].twinx()
        ax_right.plot(log["time"], log["fitness"], color="r")
        ax_right.set_ylabel("Cumulative Reward", color="r")
        ax_right.tick_params(axis="y", labelcolor="r")

        # Set axis labels
        for ax in axes.flat:
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Arb.")

        plt.tight_layout()
        plt.show()

        self.logger = None

    def evolve(self):
        """Run evolutionary learning process"""
        best_rnn = []
        best_fitness = -np.inf

        too_low_counter = 0
        too_high_counter = 0
        adaptation_threshold = 10

        for gg in range(self.num_generations):
            fitnesses = np.array(
                [
                    self.evaluate(individual, seed=gg, render=False)
                    for individual in tqdm(self.population, desc="Evaluating")
                ]
            )
            best_idx = np.argmax(fitnesses)
            worst_idx = np.argmin(fitnesses)

            # Adapt mutation rate
            if best_rnn == self.population[best_idx]:
                too_low_counter = 0
                too_high_counter += 1
            else:
                too_low_counter += 1
                too_high_counter = 0
            if too_low_counter > adaptation_threshold:
                self.mutation_rate *= 1.1
                too_low_counter = 0
            if too_high_counter > adaptation_threshold:
                self.mutation_rate *= 0.9
                too_high_counter = 0

            if fitnesses[best_idx] > best_fitness:
                best_fitness = fitnesses[best_idx]
                best_rnn = self.population[best_idx]

            print(
                f"Generation {gg+1}, Best: {fitnesses[best_idx]:.2f}, Worst: {fitnesses[worst_idx]:.2f}"
            )

            # Select top individuals
            sorted_indices = np.argsort(fitnesses)[::-1]
            self.population = [
                self.population[i] for i in sorted_indices[: self.num_individuals // 2]
            ]

            # Mutate top performers to create offspring
            for ii in range(len(self.population)):
                if np.random.rand() >= 0.5:
                    parent1 = np.random.choice(self.population)
                    parent2 = np.random.choice(self.population)
                    child = RNN.recombine(parent1, parent2)
                else:
                    parent = self.population[ii]
                    child = parent.mutate(beta_from_mean(self.mutation_rate))
                self.population.append(child)

            if gg % 10 == 0:
                self.evaluate(best_rnn, seed=0, render=False, log=True)
                self.plot()
            if gg % 100 == 0:
                file = f"../models/best_rnn_gen_{gg}_GA.pkl"
            else:
                file = f"../models/best_rnn_gen_curr_GA.pkl"
            with open(file, "wb") as f:
                pickle.dump(best_rnn, f)

        return best_rnn


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
if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    reacher = EvolveSequentialReacher(
        target_duration=(4, 2, 6),
        num_targets=10,
        num_individuals=100,
        num_generations=1000,
        num_hidden_units=25,
        mutation_rate=0.1,
        activation=tanh,
        tau=10e-3,
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
gen_idx = 500  # Specify the generation index you want to plot
model_file = f"best_rnn_gen_{gen_idx}.pkl"
model_file = "best_rnn_gen_curr.pkl"
with open(os.path.join(models_dir, model_file), "rb") as f:
    best_rnn = pickle.load(f)
reacher.render(best_rnn)

# %%
"""
.########.########..######..########..######.
....##....##.......##....##....##....##....##
....##....##.......##..........##....##......
....##....######....######.....##.....######.
....##....##.............##....##..........##
....##....##.......##....##....##....##....##
....##....########..######.....##.....######.
"""
plt.figure()
a = beta_from_mean(0.05, 5, 10000)
print(a.mean())
plt.hist(a, bins=30)
plt.show()
