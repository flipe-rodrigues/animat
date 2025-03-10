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
import jax
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
from mujoco import mjx

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
    lambda_ = 1 / tau  # Convert mean to lambda
    kernel = lambda_ * np.exp(-lambda_ * time)  # Exponential function
    return kernel / kernel.sum()  # Normalize


def truncated_exponential(mu, a, b, size=1):
    """Sample from a truncated exponential distribution using inverse CDF method."""
    lambda_ = 1 / mu
    U = np.random.uniform(0, 1, size)
    if size == 1:
        U = U[0]
    exp_a, exp_b = np.exp(-lambda_ * a), np.exp(-lambda_ * b)
    return -np.log((1 - U) * (exp_a - exp_b) + exp_b) / lambda_


def sample_entropy(samples, base=2):
    """Compute entropy directly from a vector of samples."""
    _, counts = np.unique(samples, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log(probs) / np.log(base))


def beta_from_mean(mu, nu=1, num_samples=1):
    alpha = mu * nu
    beta_param = (1 - mu) * nu
    return beta.rvs(alpha, beta_param, size=num_samples)


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
        # self.b_ctx = xavier_init(n_in=self.context_size, n_out=self.hidden_size)
        # self.b_fbk = xavier_init(n_in=self.feedback_size, n_out=self.hidden_size)
        # self.b_h = xavier_init(n_in=self.hidden_size, n_out=self.hidden_size)
        # self.b_out = xavier_init(n_in=self.hidden_size, n_out=self.output_size)
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

    def step(self, ctx, fbk):
        """Compute one RNN step"""
        ctx = (1 - self.alpha) * self.ctx + self.alpha * self.phi(ctx + self.b_ctx)
        fbk = (1 - self.alpha) * self.fbk + self.alpha * self.phi(fbk + self.b_fbk)
        h = (1 - self.alpha) * self.h + self.alpha * self.phi(
            self.W_ctx @ ctx + self.W_fbk @ fbk + self.W_h @ self.h + self.b_h
        )
        out = (1 - self.alpha) * self.out + self.alpha * logistic(
            self.W_out @ h + self.b_out
        )
        self.ctx = ctx
        self.fbk = fbk
        self.h = h
        self.out = out
        return out

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
    def __init__(
        self, model_path="../mujoco/arm_model.xml", targets_path="../src/targets.csv"
    ):
        """Initialize Mujoco simulation"""
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.mxj_model = mjx.put_model(self.model)
        self.mjx_data = mjx.put_data(self.model, self.data)

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

    def sample_target(self, num_samples=1):
        idcs2d = self.targets[0]
        x_edges = self.targets[1]
        y_edges = self.targets[2]
        sampled_idcs = idcs2d[
            np.random.choice(idcs2d.shape[0], num_samples, replace=False)
        ]
        sampled_x = x_edges[sampled_idcs[:, 0]]
        sampled_y = y_edges[sampled_idcs[:, 1]]
        target_pos = [sampled_x[0], 0, sampled_y[0]]
        self.data.mocap_pos = target_pos
        mujoco.mj_forward(self.model, self.data)
        return target_pos

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
        k = 100  # Adjust the value of k as needed
        sensor_data[
            self.num_sensors // 3 * 2 :
        ] /= k  # Assuming force sensors are in the last third
        return sensor_data

    def get_pos(self, geom_name):
        """Return current position of the end effector"""
        geom_id = self.model.geom(geom_name).id
        return self.data.geom_xpos[geom_id][:].copy()

    def distance(self, pos1, pos2):
        return np.sqrt(np.sum((pos1 - pos2) ** 2))


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
        mutation_rate,
        activation,
        tau,
    ):
        self.target_duration = target_duration
        self.num_targets = num_targets
        self.num_individuals = num_individuals
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.env = MuJoCoPlant()
        self.population = [
            RNN(
                context_size=3,
                feedback_size=self.env.num_sensors,
                hidden_size=25,
                output_size=self.env.num_actuators,
                activation=activation,
                alpha=self.env.model.opt.timestep / tau,
            )
            for _ in range(num_individuals)
        ]

    def reward(self, target_position):
        return -self.env.distance(self.env.get_pos("hand"), target_position)

    def evaluate(self, rnn, seed=123):
        """Evaluate fitness of a given RNN policy"""
        np.random.seed(seed)

        rnn.init_state()
        self.env.reset()
        target_position = self.env.sample_target()
        target_onset_time = 0
        target_duration = truncated_exponential(
            mu=self.target_duration[0],
            a=self.target_duration[1],
            b=self.target_duration[2],
        )
        trial_duration = target_duration
        total_reward = 0

        mj_model = self.env.model
        mj_data = self.env.data
        mjx_model = self.env.mxj_model

        jit_step = jax.jit(mjx.step)
        mujoco.mj_resetData(mj_model, mj_data)
        mjx_data = mjx.put_data(mj_model, mj_data)
        
        target_idx = 0
        while target_idx < self.num_targets:
            mjx_data = jit_step(mjx_model, mjx_data)
            mj_data = mjx.get_data(mj_model, mjx_data)

            sensory_feedback = self.env.get_obs()
            muscle_activations = rnn.step(target_position, sensory_feedback)
            self.env.step(muscle_activations)
            total_reward += self.reward(target_position)
            if self.env.data.time - target_onset_time >= target_duration:
                target_position = self.env.sample_target()
                target_onset_time = self.env.data.time
                target_duration = truncated_exponential(
                    mu=self.target_duration[0],
                    a=self.target_duration[1],
                    b=self.target_duration[2],
                )
                trial_duration += target_duration
                target_idx += 1

        average_reward = total_reward / trial_duration
        return average_reward

    def render(self, rnn, seed=0):
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
        progress_data = []
        fitness_data = []

        with mujoco.viewer.launch_passive(self.env.model, self.env.data) as viewer:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
            viewer.cam.lookat[:] = [0, -1.5, -0.5]
            viewer.cam.azimuth = 90
            viewer.cam.elevation = 0
            viewer.sync()

            rnn.init_state()
            self.env.reset()
            target_position = self.env.sample_target()
            target_onset_time = 0
            target_duration = truncated_exponential(
                mu=self.target_duration[0],
                a=self.target_duration[1],
                b=self.target_duration[2],
            )
            trial_duration = target_duration
            total_reward = 0
            distance_prev = self.env.distance(self.env.get_pos("hand"), target_position)

            target_idx = 0
            while viewer.is_running() and target_idx < self.num_targets:
                step_start = time.time()

                sensory_feedback = self.env.get_obs()
                muscle_activations = rnn.step(target_position, sensory_feedback)
                self.env.step(muscle_activations)
                distance = self.env.distance(self.env.get_pos("hand"), target_position)
                energy = np.mean(muscle_activations)
                progress = (distance_prev - distance) / self.env.model.opt.timestep
                reward = self.reward(target_position)
                total_reward += reward

                distance_prev = distance
                if self.env.data.time - target_onset_time >= target_duration:
                    target_position = self.env.sample_target()
                    target_onset_time = self.env.data.time
                    target_duration = truncated_exponential(
                        mu=self.target_duration[0],
                        a=self.target_duration[1],
                        b=self.target_duration[2],
                    )
                    trial_duration += target_duration
                    target_idx += 1
                    distance_prev = self.env.distance(
                        self.env.get_pos("hand"), target_position
                    )

                viewer.sync()

                time_data.append(self.env.data.time)
                for i, key in enumerate(sensor_data.keys()):
                    sensor_data[key].append(sensory_feedback[i])

                distance_data.append(distance)
                energy_data.append(energy)
                progress_data.append(progress)
                fitness_data.append(total_reward)
                target_data.append(target_position)

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = self.env.model.opt.timestep - (
                    time.time() - step_start
                )
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

            viewer.close()

            # Plot data
            fig, axes = plt.subplots(2, 2)

            # Length
            axes[0, 0].plot(time_data, sensor_data["deltoid_length"], label="Deltoid")
            axes[0, 0].plot(
                time_data, sensor_data["latissimus_length"], label="Latissimus"
            )
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
            axes[1, 0].plot(
                time_data, sensor_data["latissimus_force"], label="Latissimus"
            )
            axes[1, 0].plot(time_data, sensor_data["biceps_force"], label="Biceps")
            axes[1, 0].plot(time_data, sensor_data["triceps_force"], label="Triceps")
            axes[1, 0].set_title("Force")

            # Fitness
            axes[1, 1].plot(time_data, distance_data, label="Distance")
            axes[1, 1].plot(time_data, progress_data, label="Progress")
            axes[1, 1].plot(time_data, energy_data, label="Energy")
            axes[1, 1].set_title("Fitness")
            axes[1, 1].set_ylim([-1.05, 1.05])
            axes[1, 1].legend(loc="lower left")

            # Create a twin axis (right y-axis)
            ax_right = axes[1, 1].twinx()
            ax_right.plot(time_data, fitness_data / trial_duration, color="r")
            ax_right.set_ylabel("Cumulative", color="r")
            ax_right.tick_params(axis="y", labelcolor="r")

            # Set axis labels
            for ax in axes.flat:
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Arb.")

            plt.tight_layout()
            plt.show()

    def evolve(self):
        """Run evolutionary learning process"""
        best_rnn = []
        best_fitness = -np.inf
        for gg in range(self.num_generations):
            fitnesses = np.array(
                [
                    self.evaluate(individual, seed=gg)
                    for individual in tqdm(self.population, desc="Evaluating")
                ]
            )
            best_idx = np.argmax(fitnesses)
            worst_idx = np.argmin(fitnesses)

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
                self.render(best_rnn)
            if gg % 100 == 0:
                file = f"../models/best_rnn_gen_{gg}.pkl"
            else:
                file = f"../models/best_rnn_gen_curr.pkl"
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
        target_duration=(3, 1, 9),
        num_targets=15,
        num_individuals=100,
        num_generations=1000,
        mutation_rate=0.1,
        activation=tanh,
        tau=25e-3,
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
