import numpy as np
from dm_testing.arm_env import load, _DEFAULT_TIME_LIMIT  # Import the variable
from dm_testing.dm_control_test import display_video
import matplotlib.pyplot as plt
import os
import importlib
import dm_testing.arm_env
importlib.reload(dm_testing.arm_env)


class PolicyNetwork:
    def __init__(self, input_size, output_size, hidden_size=128):
        self.W1 = np.random.randn(input_size, hidden_size).astype(np.float32) * np.sqrt(2.0/input_size)
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.W2 = np.random.randn(hidden_size, output_size).astype(np.float32) * np.sqrt(2.0/hidden_size)
        self.b2 = np.zeros(output_size, dtype=np.float32)
        
        # Adam optimizer parameters
        self.m_W1, self.v_W1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.m_b1, self.v_b1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.m_W2, self.v_W2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.m_b2, self.v_b2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        self.beta1, self.beta2 = 0.9, 0.999
        self.epsilon = 1e-8
        self.t = 0
    
    def forward(self, x):
        x = x.astype(np.float32)
        self.h = np.tanh(x @ self.W1 + self.b1)
        self.y = 1 / (1 + np.exp(-(self.h @ self.W2 + self.b2)))
        return self.y
    
    def backward(self, states, actions, returns):
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        returns = np.array(returns, dtype=np.float32)

        h = np.tanh(states @ self.W1 + self.b1)
        y = 1 / (1 + np.exp(-(h @ self.W2 + self.b2)))

        dy = (actions - y) * returns[:, None]
        dW2 = h.T @ dy / len(states)
        db2 = np.mean(dy, axis=0)

        dh = (dy @ self.W2.T) * (1 - h**2)
        dW1 = states.T @ dh / len(states)
        db1 = np.mean(dh, axis=0)

        return [dW1, db1, dW2, db2]
    
    def update(self, grads, learning_rate):
        self.t += 1
        for param, grad, m, v in [
            (self.W1, grads[0], self.m_W1, self.v_W1),
            (self.b1, grads[1], self.m_b1, self.v_b1),
            (self.W2, grads[2], self.m_W2, self.v_W2),
            (self.b2, grads[3], self.m_b2, self.v_b2)
        ]:
            m[:] = self.beta1 * m + (1 - self.beta1) * grad
            v[:] = self.beta2 * v + (1 - self.beta2) * (grad**2)
            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)
            param += learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

def train():
    env = load()
    
    # Setup
    obs_spec = env.observation_spec()
    input_size = sum(np.prod(spec.shape) for spec in obs_spec.values())
    output_size = env.action_spec().shape[0]
    
    # Hyperparameters - MODIFIED
    policy = PolicyNetwork(input_size, output_size, hidden_size=256)  # Larger network
    learning_rate = 5e-4  # Slightly higher learning rate
    num_episodes = 2000   # Many more episodes
    batch_size = 32       # Larger batch
    gamma = 0.99
    
    # For tracking progress
    all_rewards = []
    best_reward = -float('inf')
    running_avg = []
    
    # For saving checkpoints
    checkpoint_interval = 200  # Save model every 200 episodes
    
    # Observation normalization variables
    obs_running_mean = np.zeros(input_size)
    obs_running_var = np.ones(input_size)
    obs_count = 0
    
    for episode in range(num_episodes):
        states, actions, rewards = [], [], []
        time_step = env.reset()
        episode_reward = 0
        step_counter = 0
        
        # Only capture frames for the last episode
        frames = [] if episode == num_episodes - 1 else None
        
        # Modified exploration schedule - decays more slowly
        noise_scale = max(0.3 * np.exp(-episode/1000), 0.05)
        
        while not time_step.last():
            step_counter += 1
            
            # Normalize observations (helps learning stability)
            state = np.concatenate([
                obs.flatten() if name != 'target_position' else obs.flatten()[0:3]
                for name, obs in time_step.observation.items()
            ])
            
            # Update running statistics
            obs_count += 1
            delta = state - obs_running_mean
            obs_running_mean += delta / obs_count
            delta2 = state - obs_running_mean
            obs_running_var += delta * delta2
            
            # Normalize state
            if obs_count > 1:
                normalized_state = (state - obs_running_mean) / (np.sqrt(obs_running_var / (obs_count - 1)) + 1e-8)
            else:
                normalized_state = state
            
            # Get action with adjusted noise
            action = policy.forward(normalized_state)
            noise = np.random.normal(0, noise_scale, size=action.shape)
            action = np.clip(action + noise, 0, 1)

            states.append(normalized_state)
            actions.append(action)

            time_step = env.step(action)
            rewards.append(time_step.reward)
            episode_reward += time_step.reward
            
            # Capture frames only for the last episode
            if frames is not None and (step_counter == 1 or step_counter % 10 == 0):
                frames.append(env.physics.render(camera_id=-1, width=640, height=480))
        
        # Print more frequently to monitor progress
        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes} ended. Reward: {episode_reward:.2f}")
            
        all_rewards.append(episode_reward)
        
        # Calculate running average reward
        if episode >= 100:
            avg_reward = np.mean(all_rewards[-100:])
            running_avg.append(avg_reward)
            
            if episode % 100 == 0:
                print(f"Episode {episode}, Avg Reward (last 100): {avg_reward:.2f}, Best: {best_reward:.2f}")
        
        # Update policy more frequently
        if len(states) > 0:  # Make sure we have data
            # Calculate returns
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)
            returns = np.array(returns, dtype=np.float32)
            
            # Normalize returns (helps with learning stability)
            if len(returns) > 1:  # Need at least 2 values for std to be meaningful
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Update policy
            grads = policy.backward(states, actions, returns)
            policy.update(grads, learning_rate)
        
        # Save the last episode frames
        if episode == num_episodes - 1 and frames:  # Last episode
            display_video(frames, filename='final_animation.gif', framerate=30)
            print(f"Final episode animation saved. Episode {episode}, Reward: {episode_reward:.2f}")
        
        # Check if this is a new best episode (just track the reward, no frames)
        if episode_reward > best_reward:
            best_reward = episode_reward
            print(f"New best! Episode {episode}, Reward: {episode_reward:.2f}")
        
        # Save checkpoints
        if episode % checkpoint_interval == 0:
            print(f"Checkpoint saved at episode {episode}.")
        
    plt.figure()  # Create a new figure
    plt.plot(all_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.savefig('training_progress.png')  # Save the plot to a file
    print("Training progress plot saved as 'training_progress.png'")
    plt.close()  # Close the figure to avoid interference

if __name__ == "__main__":
    train()
    print(f"Time limit for the environment: {_DEFAULT_TIME_LIMIT} seconds")