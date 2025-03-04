#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import mujoco
import mujoco.viewer
from tqdm import tqdm

#%%

# RNN Controller
class RNN:
    def __init__(self, input_size=15, hidden_size=100, output_size=4):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize RNN weights
        self.W_ih = np.random.randn(hidden_size, input_size) * 0.1
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
        self.W_ho = np.random.randn(output_size, hidden_size) * 0.1
        self.h = np.zeros(hidden_size)  # Hidden state

    def reset(self):
        """Reset hidden state between episodes"""
        self.h = np.zeros(self.hidden_size)

    def step(self, input):
        """Compute one RNN step"""
        self.h = 1 / (1 + np.exp(-np.dot(self.W_ih, input) - np.dot(self.W_hh, self.h)))  # Sigmoid activation for hidden state
        output = 1 / (1 + np.exp(-np.dot(self.W_ho, self.h)))  # Sigmoid activation for output
        return output

# Mujoco Environment for 2-Joint Limb
class Limb2DMujoco:
    def __init__(self, model_path="mujoco/arm_model.xml", targets_path="src/targets.csv"):
        """Initialize Mujoco simulation"""
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.targets = pd.read_csv(targets_path).values

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
    
    def get_pos(self):
        """Return current position of the end effector"""
        hand_id = self.model.geom('hand').id
        return self.data.geom_xpos[hand_id][:].copy()

    def sample_target(self):
        """Read a random target position from targets.csv"""
        random_index = np.random.randint(len(self.targets))
        target_pos = self.targets[random_index]
        self.data.mocap_pos = target_pos
        return target_pos
    
# Evolutionary Strategy (ES) Optimization
class EvolutionaryLearner:
    def __init__(self, trial_dur, num_trials, num_individuals, num_generations, mutation_rate):
        self.trial_dur = trial_dur
        self.num_trials = num_trials
        self.num_individuals = num_individuals
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.population = [self.init_params() for _ in range(num_individuals)]

    def init_params(self):
        """Initialize RNN parameters"""
        return np.random.randn(25 * (15 + 25 + 4)) * 0.1

    def evaluate(self, individual, render=False):
        """Evaluate fitness of a given RNN policy"""
        
        rnn = self.params_to_rnn(self.population[individual])
        rnn.reset()

        # progress = (individual + 1) / self.num_individuals
        # print(f"Progress: {progress:.2%}")

        env = Limb2DMujoco()

        total_reward = 0
                
        for trial in range(self.num_trials):
            env.reset()
            target_pos = env.sample_target()

            if render == True and trial == 0 and individual == 0:
                with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
                    viewer.cam.lookat[:] = [0, 0, -.5]
                    viewer.cam.azimuth = 90
                    viewer.cam.elevation = 0
                    viewer.sync()

                    while viewer.is_running():
                        while env.data.time < self.trial_dur:
                            step_start = time.time()

                            sensory_feedback = env.get_obs()
                            input_vec = np.concatenate([target_pos, sensory_feedback])
                            muscle_activations = rnn.step(input_vec)
                            env.step(muscle_activations)
                            distance = np.linalg.norm(env.get_pos() - target_pos)
                            total_reward -= distance / self.num_trials
                                
                            viewer.sync()

                            # Rudimentary time keeping, will drift relative to wall clock.
                            time_until_next_step = env.model.opt.timestep - (time.time() - step_start)
                            if time_until_next_step > 0:
                                time.sleep(time_until_next_step)

                        viewer.close()
            else:
                while env.data.time < self.trial_dur:
                    sensory_feedback = env.get_obs()
                    input_vec = np.concatenate([target_pos, sensory_feedback])
                    muscle_activations = rnn.step(input_vec)
                    env.step(muscle_activations)
                    distance = np.linalg.norm(env.get_pos() - target_pos)
                    total_reward -= distance / self.num_trials

        return total_reward

    def params_to_rnn(self, params):
        """Convert flat parameters into RNN weight matrices"""
        rnn = RNN(15, 25, 4)
        idx = 0
        size_Wih = rnn.hidden_size * rnn.input_size
        size_Whh = rnn.hidden_size * rnn.hidden_size
        size_Who = rnn.output_size * rnn.hidden_size

        rnn.W_ih = params[idx:idx + size_Wih].reshape(rnn.hidden_size, rnn.input_size)
        idx += size_Wih
        rnn.W_hh = params[idx:idx + size_Whh].reshape(rnn.hidden_size, rnn.hidden_size)
        idx += size_Whh
        rnn.W_ho = params[idx:idx + size_Who].reshape(rnn.output_size, rnn.hidden_size)

        return rnn

    def evolve(self):
        """Run evolutionary learning process"""
        for gen in range(self.num_generations):
            fitnesses = np.array([self.evaluate(i, render=True) for i in tqdm(range(self.num_individuals), desc="Evaluating")])
            best_idx = np.argmax(fitnesses)
            worst_idx = np.argmin(fitnesses)

            print(f"Generation {gen+1}, Best Fitness: {fitnesses[best_idx]:.4f}, Worst Fitness: {fitnesses[worst_idx]:.4f}")

            # Select top individuals
            sorted_indices = np.argsort(fitnesses)[::-1]
            self.population = [self.population[i] for i in sorted_indices[:self.num_individuals // 2]]

            # Mutate top performers to create offspring
            new_population = []
            for individual in self.population:
                for _ in range(2):  # Each parent creates 2 offspring
                    child = individual + np.random.randn(len(individual)) * self.mutation_rate
                    new_population.append(child)

            self.population = new_population[:self.num_individuals]

        return self.population[0]  # Return best evolved parameters

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
    learner = EvolutionaryLearner(trial_dur=3, num_trials=10, num_individuals=50, num_generations=100, mutation_rate=.1)
    best_params = learner.evolve()

#%%
# Plot RNN weights as heatmaps
def plot_weights(rnn):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rnn.W_ih, aspect='auto', cmap='viridis')
    axes[0].set_title('Input to Hidden Weights')
    axes[0].set_xlabel('Input Units')
    axes[0].set_ylabel('Hidden Units')

    axes[1].imshow(rnn.W_hh, aspect='auto', cmap='viridis')
    axes[1].set_title('Hidden to Hidden Weights')
    axes[1].set_xlabel('Hidden Units')
    axes[1].set_ylabel('Hidden Units')

    axes[2].imshow(rnn.W_ho, aspect='auto', cmap='viridis')
    axes[2].set_title('Hidden to Output Weights')
    axes[2].set_xlabel('Hidden Units')
    axes[2].set_ylabel('Output Units')

    plt.tight_layout()
    plt.show()

# Example usage
best_rnn = learner.params_to_rnn(best_params)
plot_weights(best_rnn)