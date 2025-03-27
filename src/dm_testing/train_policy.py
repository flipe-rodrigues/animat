import numpy as np
from dm_testing.arm_env import load
from dm_testing.dm_control_test import display_video
import matplotlib.pyplot as plt
import os

class PolicyNetwork:
    def __init__(self, input_size, output_size, hidden_size=128):
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0/input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0/hidden_size)
        self.b2 = np.zeros(output_size)
        
        # Adam optimizer parameters
        self.m_W1, self.v_W1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.m_b1, self.v_b1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.m_W2, self.v_W2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.m_b2, self.v_b2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        self.beta1, self.beta2 = 0.9, 0.999
        self.epsilon = 1e-8
        self.t = 0
    
    def forward(self, x, training=False):
        # Store inputs and compute hidden layer
        self.x = x
        self.h = np.tanh(x @ self.W1 + self.b1)
        # Use sigmoid for output to ensure [0,1] range
        self.y = 1/(1 + np.exp(-(self.h @ self.W2 + self.b2)))
        return self.y
    
    def backward(self, states, actions, returns):
        dW1, db1 = np.zeros_like(self.W1), np.zeros_like(self.b1)
        dW2, db2 = np.zeros_like(self.W2), np.zeros_like(self.b2)
        
        for state, action, R in zip(states, actions, returns):
            # Forward pass
            h = np.tanh(state @ self.W1 + self.b1)
            y = 1/(1 + np.exp(-(h @ self.W2 + self.b2)))
            
            # Proper log-probability gradient for continuous actions
            # Treating policy output as mean of a Gaussian with fixed variance
            std = 0.1  # Fixed standard deviation
            log_prob_grad = (action - y) / (std**2)
            dy = log_prob_grad * R
            
            dW2 += np.outer(h, dy)
            db2 += dy
            
            dh = (dy @ self.W2.T) * (1 - h**2)
            dW1 += np.outer(state, dh)
            db1 += dh
        
        return [dW1/len(states), db1/len(states), 
                dW2/len(states), db2/len(states)]
    
    def update(self, grads, learning_rate):
        self.t += 1
        for param, grad, m, v in [
            (self.W1, grads[0], self.m_W1, self.v_W1),
            (self.b1, grads[1], self.m_b1, self.v_b1),
            (self.W2, grads[2], self.m_W2, self.v_W2),
            (self.b2, grads[3], self.m_b2, self.v_b2)
        ]:
            # Adam update
            m[:] = self.beta1 * m + (1 - self.beta1) * grad
            v[:] = self.beta2 * v + (1 - self.beta2) * (grad**2)
            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)
            param += learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

def train():
    env = load()
    
    # Setup
    obs_spec = env.observation_spec()
    input_size = sum(spec.shape[0] if len(spec.shape) > 0 else 1 
                    for spec in obs_spec.values())
    output_size = env.action_spec().shape[0]
    
    # Hyperparameters
    policy = PolicyNetwork(input_size, output_size)
    learning_rate = 3e-4
    num_episodes = 3000  # Increase from 1000
    batch_size = 4  
    gamma = 0.99
    
    all_rewards = []
    best_reward = -float('inf')
    
    for episode in range(num_episodes):
        states, actions, rewards = [], [], []
        time_step = env.reset()
        episode_reward = 0
        
        # Record video for first episode and new best episodes
        record_video = episode == 0 or (len(all_rewards) > 0 and episode_reward > best_reward)
        frames = [] if record_video else None
        
        while not time_step.last():
            state = np.concatenate([obs.flatten() for obs in time_step.observation.values()])
            
            # Get action with noise
            action = policy.forward(state)
            noise = np.random.normal(0, max(0.3 * np.exp(-episode/500), 0.05), size=action.shape)
            action = np.clip(action + noise, 0, 1)
            
            states.append(state)
            actions.append(action)
            
            time_step = env.step(action)
            rewards.append(time_step.reward)
            episode_reward += time_step.reward
            
            if frames is not None:
                frames.append(env.physics.render())
        
        if frames is not None and episode_reward > best_reward:
            best_reward = episode_reward
            display_video(frames, framerate=30)
            print(f"New best! Episode {episode}, Reward: {episode_reward:.2f}")
        
        all_rewards.append(episode_reward)
        
        # Update policy
        if (episode + 1) % batch_size == 0:
            # Calculate returns
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)
            returns = np.array(returns)

            # Use a baseline for advantage estimation
            baseline = np.mean(returns)
            advantages = returns - baseline

            # Normalize advantages
            if len(advantages) > 1 and advantages.std() > 0:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Update policy (use advantages instead of returns)
            grads = policy.backward(states, actions, advantages)
            policy.update(grads, learning_rate)
        
        if episode % 10 == 0:
            avg_reward = np.mean(all_rewards[-100:]) if len(all_rewards) >= 100 else np.mean(all_rewards)
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Best: {best_reward:.2f}")

if __name__ == "__main__":
    train()