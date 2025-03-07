#%%
"""
.####.##.....##.########...#######..########..########..######.
..##..###...###.##.....##.##.....##.##.....##....##....##....##
..##..####.####.##.....##.##.....##.##.....##....##....##......
..##..##.###.##.########..##.....##.########.....##.....######.
..##..##.....##.##........##.....##.##...##......##..........##
..##..##.....##.##........##.....##.##....##.....##....##....##
.####.##.....##.##.........#######..##.....##....##.....######.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import time
import mujoco
import mujoco.viewer
from tqdm import tqdm

#%%
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
    def __init__(self, context_size, feedback_size, hidden_size, output_size):
        self.context_size = context_size
        self.feedback_size = feedback_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.init_weights()
        self.init_biases()
        self.init_state()

    def init_weights(self):
        self.W_ctx = np.random.randn(self.hidden_size, self.context_size) * np.sqrt(1 / self.context_size)
        self.W_fbk = np.random.randn(self.hidden_size, self.feedback_size) * np.sqrt(1 / self.feedback_size)
        self.W_h = np.random.randn(self.hidden_size, self.hidden_size) * np.sqrt(1 / self.hidden_size)
        self.W_out = np.random.randn(self.output_size, self.hidden_size) * np.sqrt(1 / self.hidden_size)
    
    def init_biases(self):
        self.b_ctx = np.zeros(self.context_size)
        self.b_fbk = np.zeros(self.feedback_size)
        self.b_h = np.zeros(self.hidden_size)
        self.b_out = np.zeros(self.output_size)
    
    def recombine(p1, p2):
        child = RNN(p1.context_size, p1.feedback_size, p1.hidden_size, p1.output_size)
        child.W_ctx = RNN.recombine_matrices(p1.W_ctx, p2.W_ctx)
        child.W_fbk = RNN.recombine_matrices(p1.W_fbk, p2.W_fbk)
        child.W_h = RNN.recombine_matrices(p1.W_h, p2.W_h)
        child.W_out = RNN.recombine_matrices(p1.W_out, p2.W_out)
        child.b_ctx = RNN.recombine_matrices(p1.b_ctx, p2.b_ctx)
        child.b_fbk = RNN.recombine_matrices(p1.b_fbk, p2.b_fbk)
        child.b_h = RNN.recombine_matrices(p1.b_h, p2.b_h)
        child.b_out = RNN.recombine_matrices(p1.b_out, p2.b_out)
        return child

    def recombine_matrices(A, B):
        mask = np.random.rand(*A.shape) > 0.5
        return np.where(mask, A, B)

    def mutate(self, rate):
        mutant = copy.deepcopy(self)
        mutant.W_ctx += np.random.randn(mutant.hidden_size, mutant.context_size) * np.sqrt(1 / mutant.context_size) * rate 
        mutant.W_fbk += np.random.randn(mutant.hidden_size, mutant.feedback_size) * np.sqrt(1 / mutant.feedback_size) * rate
        mutant.W_h += np.random.randn(mutant.hidden_size, mutant.hidden_size) * np.sqrt(1 / mutant.hidden_size) * rate
        mutant.W_out += np.random.randn(mutant.output_size, mutant.hidden_size) * np.sqrt(1 / mutant.hidden_size) * rate
        mutant.b_h += np.random.randn(mutant.hidden_size) * rate
        mutant.b_out += np.random.randn(mutant.output_size) * rate
        return mutant

    def init_state(self):
        """Reset hidden state between episodes"""
        self.h = np.zeros(self.hidden_size) + .5

    def get_params(self):
        """Convert RNN weight matrices into flat parameters"""
        return np.concatenate([self.W_ctx.flatten(), self.W_fbk.flatten(), self.W_h.flatten(), self.W_out.flatten()])

    def step(self, ctx, fbk):
        """Compute one RNN step"""
        self.h = self.logistic(self.W_ctx @ ctx + self.W_fbk @ fbk + self.W_h @ self.h + self.b_h)
        output = self.logistic(self.W_out @ self.h + self.b_out)
        return output

    def logistic(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)
    
    def relu(self, x):
        return np.maximum(0, x)

#%%
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
    def __init__(self, model_path="../mujoco/arm_model.xml", targets_path="../src/targets.csv"):
        """Initialize Mujoco simulation"""
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.num_sensors = self.model.nsensor
        self.num_actuators = self.model.nu
        self.parse_targets()

    def parse_targets(self, targets_path="../src/targets.csv"):
        targets = pd.read_csv(targets_path).values
        H = np.histogram2d(targets[:,0], targets[:,2], bins=100)
        self.targets = []
        self.targets.append(np.argwhere(H[0] > 0))
        self.targets.append(H[1])
        self.targets.append(H[2])

    def sample_target(self, num_samples=1):
        idcs2d = self.targets[0]
        x_edges = self.targets[1]
        y_edges = self.targets[2]
        sampled_idcs = idcs2d[np.random.choice(idcs2d.shape[0], num_samples, replace=False)]
        sampled_x = x_edges[sampled_idcs[:, 0]]
        sampled_y = y_edges[sampled_idcs[:, 1]]
        target_pos = [sampled_x[0], 0, sampled_y[0]]
        self.data.mocap_pos = target_pos
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
        sensor_data[self.num_sensors//3*2:] /= k  # Assuming force sensors are in the last third
        return sensor_data
    
    def get_pos(self, geom_name):
        """Return current position of the end effector"""
        geom_id = self.model.geom(geom_name).id
        return self.data.geom_xpos[geom_id][:].copy()
    
    def distance(self, pos1, pos2):
        return np.sqrt(np.sum((pos1 - pos2) ** 2))

#%%
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
    def __init__(self, trial_dur, num_targets, num_individuals, num_generations, mutation_rate):
        self.trial_dur = trial_dur
        self.num_targets = num_targets
        self.num_individuals = num_individuals
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.env = MuJoCoPlant()
        self.population = [RNN(3, self.env.num_sensors, 100, self.env.num_actuators) for _ in range(num_individuals)]

    def evaluate(self, rnn, gen_idx):
        """Evaluate fitness of a given RNN policy"""
        rnn.init_state()
        total_reward = 0
   
        np.random.seed(gen_idx)

        for trial in range(self.num_targets):
            self.env.reset()
            target_pos = self.env.sample_target()

            while self.env.data.time < self.trial_dur:
                sensory_feedback = self.env.get_obs()
                muscle_activations = rnn.step(target_pos, sensory_feedback)
                self.env.step(muscle_activations)
                distance = self.env.distance(self.env.get_pos('hand'), target_pos)
                total_reward -= distance

        average_reward = total_reward / self.trial_dur / self.num_targets

        return average_reward

    def render(self, rnn, num_trials):
        """Render a couple of trials for a set of RNN params"""
        rnn.init_state()
        total_reward = 0

        for trial in range(min(num_trials, self.num_targets)):
            self.env.reset()
            target_pos = self.env.sample_target()

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
            distance_data = []
            
            with mujoco.viewer.launch_passive(self.env.model, self.env.data) as viewer:
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
                viewer.cam.lookat[:] = [0, -1.5, -.5]
                viewer.cam.azimuth = 90
                viewer.cam.elevation = 0
                viewer.sync()

                while viewer.is_running() and self.env.data.time < self.trial_dur:
                    step_start = time.time()

                    sensory_feedback = self.env.get_obs()
                    muscle_activations = rnn.step(target_pos, sensory_feedback)
                    self.env.step(muscle_activations)
                    
                    viewer.sync()

                    time_data.append(self.env.data.time)
                    for i, key in enumerate(sensor_data.keys()):
                        sensor_data[key].append(sensory_feedback[i])

                    distance = self.env.distance(self.env.get_pos('hand'), target_pos)
                    total_reward -= distance
                    distance_data.append(distance)

                    # Rudimentary time keeping, will drift relative to wall clock.
                    time_until_next_step = self.env.model.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)

                viewer.close()

                # Plot data
                fig, axes = plt.subplots(2, 2)

                # deltoid
                axes[0, 0].plot(time_data, sensor_data["deltoid_length"], label="Deltoid")
                axes[0, 0].plot(time_data, sensor_data["latissimus_length"], label="Latissimus")
                axes[0, 0].plot(time_data, sensor_data["biceps_length"], label="Biceps")
                axes[0, 0].plot(time_data, sensor_data["triceps_length"], label="Triceps")
                axes[0, 0].set_title("Length")
                axes[0, 0].legend()

                # latissimus
                axes[0, 1].plot(time_data, sensor_data["deltoid_velocity"], label="Deltoid")
                axes[0, 1].plot(time_data, sensor_data["latissimus_velocity"], label="Latissimus")
                axes[0, 1].plot(time_data, sensor_data["biceps_velocity"], label="Biceps")
                axes[0, 1].plot(time_data, sensor_data["triceps_velocity"], label="Triceps")
                axes[0, 1].set_title("Velocity")
                axes[0, 1].legend()

                # Biceps
                axes[1, 0].plot(time_data, sensor_data["deltoid_force"], label="Deltoid")
                axes[1, 0].plot(time_data, sensor_data["latissimus_force"], label="Latissimus")
                axes[1, 0].plot(time_data, sensor_data["biceps_force"], label="Biceps")
                axes[1, 0].plot(time_data, sensor_data["triceps_force"], label="Triceps")
                axes[1, 0].set_title("Force")
                axes[1, 0].legend()

                # Biceps
                axes[1, 1].plot(time_data, distance_data)
                axes[1, 1].set_title("Distance to target")
                axes[1, 1].set_ylim([0, 1])

                # Set axis labels
                for ax in axes.flat:
                    ax.set_xlabel("Steps")
                    ax.set_ylabel("arb.")

                plt.tight_layout()
                plt.show()

    def evolve(self):
        """Run evolutionary learning process"""
        best_rnn = []
        best_fitness = -np.inf
        for gg in range(self.num_generations):
            fitnesses = np.array([self.evaluate(individual, gg) for individual in tqdm(self.population, desc="Evaluating")])
            best_idx = np.argmax(fitnesses)
            worst_idx = np.argmin(fitnesses)

            if fitnesses[best_idx] > best_fitness:
                best_fitness = fitnesses[best_idx]
                best_rnn = self.population[best_idx]

            print(f"Generation {gg+1}, Best Fitness: {fitnesses[best_idx]:.2f}, Worst Fitness: {fitnesses[worst_idx]:.2f}")

            # Select top individuals
            sorted_indices = np.argsort(fitnesses)[::-1]
            self.population = [self.population[i] for i in sorted_indices[:self.num_individuals // 2]]

            # Mutate top performers to create offspring
            for ii in range(len(self.population)):
                if np.random.rand() >= .5:
                    parent1 = np.random.choice(self.population)
                    parent2 = np.random.choice(self.population)
                    child = RNN.recombine(parent1, parent2).mutate(self.mutation_rate)
                else:
                    parent = self.population[ii]
                    child = parent.mutate(self.mutation_rate)
                self.population.append(child)

            # self.render(best_rnn, num_trials=10) 

        return best_rnn

#%%
"""
.##.....##....###....####.##....##
.###...###...##.##....##..###...##
.####.####..##...##...##..####..##
.##.###.##.##.....##..##..##.##.##
.##.....##.#########..##..##..####
.##.....##.##.....##..##..##...###
.##.....##.##.....##.####.##....##
"""
if __name__ == "__main__":
    reacher = EvolveSequentialReacher(
        trial_dur=3, 
        num_targets=10, 
        num_individuals=100,
        num_generations=300, 
        mutation_rate=.15)
    
    best_rnn = reacher.evolve()

#%%
reacher.render(best_rnn,num_trials=3)

#%%
x = reacher.env.targets[:,0]
y = reacher.env.targets[:,2]

# Create a 2D histogram
# H = plt.hist2d(x, y, bins=30, cmap='Blues')
H = np.histogram2d(x, y, bins=100)
counts = H[0]

# Find the indices where counts are equal to 1
indices = np.argwhere(counts > 0)

# Uniformly sample from these indices
num_samples = 1000  # Adjust the number of samples as needed
sampled_indices = indices[np.random.choice(indices.shape[0], num_samples, replace=False)]

# Convert the sampled indices back to x, y coordinates
x_edges = H[1]
y_edges = H[2]
sampled_x = x_edges[sampled_indices[:, 0]]
sampled_y = y_edges[sampled_indices[:, 1]]

# Plot the sampled points
plt.scatter(sampled_x, sampled_y, color='red', marker='o', label='Sampled Points')
# plt.imshow(counts, aspect='auto', cmap='viridis')

# Add a colorbar
plt.colorbar(label='Frequency')

# Labels and title
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("2D Histogram")

# Show plot
plt.show()

# plt.plot(x,y,marker='.',markersize=3)