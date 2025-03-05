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
        self.reset_state()

    def from_params(self, params):
        rnn = RNN(self.context_size, self.feedback_size, self.hidden_size, self.output_size)
        idx = 0
        rnn.W_ctx = params[idx:idx + rnn.W_ctx.size].reshape(rnn.hidden_size, rnn.context_size)
        idx += rnn.W_ctx.size
        rnn.W_fbk = params[idx:idx + rnn.W_fbk.size].reshape(rnn.hidden_size, rnn.feedback_size)
        idx += rnn.W_fbk.size
        rnn.W_rnn = params[idx:idx + rnn.W_rnn.size].reshape(rnn.hidden_size, rnn.hidden_size)
        idx += rnn.W_rnn.size
        rnn.W_out = params[idx:idx + rnn.W_out.size].reshape(rnn.output_size, rnn.hidden_size)
        return rnn

    def init_weights(self):
        self.W_ctx = np.random.randn(self.hidden_size, self.context_size)
        self.W_fbk = np.random.randn(self.hidden_size, self.feedback_size)
        self.W_rnn = np.random.randn(self.hidden_size, self.hidden_size)
        self.W_out = np.random.randn(self.output_size, self.hidden_size)
    
    def mutate_weights(self, rate):
        self.W_ctx += np.random.randn(self.hidden_size, self.context_size) * rate
        self.W_fbk += np.random.randn(self.hidden_size, self.feedback_size) * rate
        self.W_rnn += np.random.randn(self.hidden_size, self.hidden_size) * rate
        self.W_out += np.random.randn(self.output_size, self.hidden_size) * rate

    def reset_state(self):
        """Reset hidden state between episodes"""
        self.h = np.zeros(self.hidden_size)

    def get_params(self):
        """Convert RNN weight matrices into flat parameters"""
        return np.concatenate([self.W_ctx.flatten(), self.W_fbk.flatten(), self.W_rnn.flatten(), self.W_out.flatten()])

    def step(self, context, feedback):
        """Compute one RNN step"""
        self.h = self.activation_fcn(self.W_ctx @ context + self.W_fbk @ feedback + self.W_rnn @ self.h)
        output = self.activation_fcn(self.W_out @ self.h)
        return output
    
    def activation_fcn(self,x):
        return self.logistic_fcn(x)

    def logistic_fcn(self, x):
        return 1 / (1 + np.exp(-x))

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
        self.targets = pd.read_csv(targets_path).values
        self.num_sensors = self.model.nsensor
        self.num_actuators = self.model.nu

    def reset(self):
        """Reset limb state"""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self.get_obs()

    def step(self, muscle_activations):
        """Apply torques and step simulation"""
        self.data.ctrl[:] = muscle_activations
        mujoco.mj_step(self.model, self.data)

    def get_obs(self):
        """Return joint angles, velocities, and end-effector position"""
        sensor_data = self.data.sensordata
        return sensor_data
    
    def get_pos(self, geom_name):
        """Return current position of the end effector"""
        geom_id = self.model.geom(geom_name).id
        return self.data.geom_xpos[geom_id][:].copy()

    def sample_target(self):
        """Read a random target position from targets.csv"""
        random_index = np.random.randint(len(self.targets))
        target_pos = self.targets[random_index]
        self.data.mocap_pos = target_pos
        return target_pos
    
    def distance(self, pos1, pos2):
        return np.linalg.norm(pos1 - pos2)

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
    def __init__(self, trial_dur, num_targets, num_individuals, num_parents, num_generations, mutation_rate):
        self.trial_dur = trial_dur
        self.num_targets = num_targets
        self.num_individuals = num_individuals
        self.num_parents = num_parents
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.env = MuJoCoPlant()
        self.population = [RNN(3, self.env.num_sensors, 25, self.env.num_actuators) for _ in range(num_individuals)]

    def evaluate(self, rnn):
        """Evaluate fitness of a given RNN policy"""
        rnn.reset_state()
        total_reward = 0
                
        for trial in range(self.num_targets):
            self.env.reset()
            target_pos = self.env.sample_target()
            initial_distance = self.env.distance(self.env.get_pos('hand'), target_pos)

            while self.env.data.time < self.trial_dur:
                sensory_feedback = self.env.get_obs()
                muscle_activations = rnn.step(target_pos, sensory_feedback)
                total_activation = sum(muscle_activations)
                self.env.step(muscle_activations)
                distance = self.env.distance(self.env.get_pos('hand'), target_pos)
                norm_distance = distance / initial_distance
                total_reward -= norm_distance + total_activation

        average_reward = total_reward / self.trial_dur / self.num_targets

        return average_reward

    def render(self, rnn, num_trials):
        """Render a couple of trials for a set of RNN params"""

        rnn.reset_state()

        for trial in range(min(num_trials, self.num_targets)):
            self.env.reset()
            target_pos = self.env.sample_target()

            with mujoco.viewer.launch_passive(self.env.model, self.env.data) as viewer:
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
                viewer.cam.lookat[:] = [0, 0, -.5]
                viewer.cam.azimuth = 90
                viewer.cam.elevation = 0
                viewer.sync()

                while viewer.is_running() and self.env.data.time < self.trial_dur:
                    step_start = time.time()

                    sensory_feedback = self.env.get_obs()
                    muscle_activations = rnn.step(target_pos, sensory_feedback)
                    self.env.step(muscle_activations)
                        
                    viewer.sync()

                    # Rudimentary time keeping, will drift relative to wall clock.
                    time_until_next_step = self.env.model.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)

                viewer.close()

    def evolve(self):
        """Run evolutionary learning process"""
        best_fitness = -np.inf
        for gen in range(self.num_generations):
            fitnesses = np.array([self.evaluate(individual) for individual in tqdm(self.population, desc="Evaluating")])
            best_idx = np.argmax(fitnesses)
            worst_idx = np.argmin(fitnesses)

            print(f"Generation {gen+1}, Best Fitness: {fitnesses[best_idx]:.4f}, Worst Fitness: {fitnesses[worst_idx]:.4f}")
            
            print(self.population)
            
            # Select top individuals
            sorted_indices = np.argsort(fitnesses)[::-1]
            self.population = [self.population[i] for i in sorted_indices[:self.num_individuals // self.num_parents]]

            if np.max(fitnesses) > best_fitness:
                best_rnn = self.population[0]   

            # Mutate top performers to create offspring
            new_population = []
            for individual in self.population:
                for _ in range(self.num_parents):
                    child = individual.mutate_weights(self.mutation_rate)
                    print(child)
                    new_population.append(child)

            self.population = new_population[:self.num_individuals]

            print(self.population)

            self.render(best_rnn, num_trials=3) 

        return best_params  # Return best evolved parameters

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
        num_individuals=10,
        num_parents = 2,
        num_generations=100, 
        mutation_rate=.1)
    best_params = reacher.evolve()
# %%
