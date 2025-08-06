"""Asatryan & Feldman (1965) Unloading Response Experiment."""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

from envs.dm_env import make_arm_env
from networks.rnn import RNNPolicy
from encoders.encoders import IdentityEncoder
from utils import encode_numpy

class AsatryanFeldmanExperiment:
    """Implement the classic unloading response experiment."""
    
    def __init__(self, model_path=None):
        """Initialize experiment with trained RNN model."""
        if model_path:
            self.policy = self.load_trained_model(model_path)
        else:
            # Create a simple test policy
            self.policy = RNNPolicy(input_dim=15, action_dim=4, hidden_size=32)
        
        self.encoder = IdentityEncoder(obs_dim=15)
        
    def load_trained_model(self, model_path):
        """Load a trained RNN policy."""
        checkpoint = torch.load(model_path, map_location='cpu')
        config = checkpoint['config']
        
        policy = RNNPolicy(
            input_dim=config.get('input_dim', 15),
            action_dim=config.get('action_dim', 4),
            hidden_size=config['hidden_size'],
            alpha=1.0/config.get('tau_mem', 10.0)
        )
        
        policy.load_state_dict(checkpoint['model_state_dict'])
        policy.eval()
        return policy
    
    def run_weight_perturbation_trial(self, target_angles=[315, 345], 
                                     weights=[0.1, 0.3, 0.5], 
                                     weight_durations=10.0,
                                     max_steps=500):
        """Run single trial with weight perturbations."""
        
        # Setup environment
        env = make_arm_env(random_seed=42)
        timestep = env.reset()
        
        # Initialize RNN
        hidden = self.policy.init_hidden(1)
        
        # Data logging
        data_log = {
            'time': [],
            'elbow_angle': [],
            'elbow_torque': [],
            'muscle_activations': [],
            'applied_weight': [],
            'target_position': [],
            'hand_position': []
        }
        
        # Weight perturbation schedule
        weight_times = np.arange(0, len(weights) * weight_durations, weight_durations)
        current_weight_idx = 0
        
        for step in range(max_steps):
            # Extract observation
            muscle_data = timestep.observation['muscle_sensors']
            target_pos = timestep.observation['target_position'].flatten()
            raw_obs = np.concatenate([muscle_data, target_pos])
            
            # Encode observation
            encoded_obs = encode_numpy(self.encoder, raw_obs)
            
            # RNN forward pass
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(encoded_obs).unsqueeze(0)
                action_tensor, hidden = self.policy(obs_tensor, hidden)
                action = action_tensor.squeeze(0).numpy()
            
            # Apply weight perturbation
            current_time = step * env.control_timestep()
            if (current_weight_idx < len(weight_times) and 
                current_time >= weight_times[current_weight_idx]):
                
                # This is where you'd apply the weight - simplified here
                applied_weight = weights[current_weight_idx]
                current_weight_idx += 1
                print(f"Applied weight: {applied_weight} kg at time {current_time:.2f}s")
            else:
                applied_weight = weights[min(current_weight_idx, len(weights)-1)]
            
            # Step environment
            timestep = env.step(action)
            
            # Log data (simplified - you'd get real measurements from physics)
            hand_pos = env.physics.bind(env._task._arm.hand).xpos
            target_pos = env.physics.bind(env._task._arm.target).mocap_pos
            
            data_log['time'].append(current_time)
            data_log['elbow_angle'].append(self.get_elbow_angle(env.physics))
            data_log['elbow_torque'].append(self.get_elbow_torque(env.physics))
            data_log['muscle_activations'].append(action.copy())
            data_log['applied_weight'].append(applied_weight)
            data_log['target_position'].append(target_pos.copy())
            data_log['hand_position'].append(hand_pos.copy())
            
            if timestep.last():
                break
        
        env.close()
        return data_log
    
    def get_elbow_angle(self, physics):
        """Extract elbow angle from physics state."""
        # This depends on your specific model structure
        # You'd need to identify the elbow joint and extract its angle
        return 0.0  # Placeholder
    
    def get_elbow_torque(self, physics):
        """Extract elbow torque from physics state."""
        # This depends on your specific model structure
        return 0.0  # Placeholder
    
    def analyze_unloading_responses(self, data_log, weight_durations=10.0):
        """Analyze unloading responses in the data."""
        
        # Convert to numpy arrays
        time = np.array(data_log['time'])
        angles = np.array(data_log['elbow_angle'])
        torques = np.array(data_log['elbow_torque'])
        weights = np.array(data_log['applied_weight'])
        muscle_acts = np.array(data_log['muscle_activations'])
        
        # Segment data by weight periods
        weight_transitions = np.where(np.diff(weights) != 0)[0] + 1
        segments = []
        
        start_idx = 0
        for transition_idx in weight_transitions:
            end_idx = transition_idx
            if end_idx - start_idx > 10:  # Minimum segment length
                segments.append({
                    'time': time[start_idx:end_idx],
                    'angles': angles[start_idx:end_idx],
                    'torques': torques[start_idx:end_idx],
                    'weight': weights[start_idx],
                    'muscle_acts': muscle_acts[start_idx:end_idx]
                })
            start_idx = end_idx
        
        # Analyze each segment for steady-state values
        results = []
        for segment in segments:
            # Take second half for steady-state analysis
            n_half = len(segment['angles']) // 2
            steady_state = {
                'weight': segment['weight'],
                'avg_angle': np.mean(segment['angles'][n_half:]),
                'avg_torque': np.mean(segment['torques'][n_half:]),
                'avg_bicep': np.mean(segment['muscle_acts'][n_half:, 2]),  # Bicep
                'avg_tricep': np.mean(segment['muscle_acts'][n_half:, 3])  # Tricep
            }
            results.append(steady_state)
        
        return results
    
    def plot_asatryan_feldman_results(self, steady_state_results, save_path=None):
        """Create the classic Asatryan & Feldman plot."""
        
        # Extract data for plotting
        angles = [r['avg_angle'] for r in steady_state_results]
        torques = [r['avg_torque'] for r in steady_state_results]
        bicep_acts = [r['avg_bicep'] for r in steady_state_results]
        tricep_acts = [r['avg_tricep'] for r in steady_state_results]
        
        # Convert angles to degrees
        angles_deg = np.degrees(angles)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot torque vs angle relationship
        ax.plot(angles_deg, torques, 'k-', linewidth=2, label='Unloading Response')
        
        # Add muscle activation as error bars or color coding
        scatter = ax.scatter(angles_deg, torques, 
                           c=bicep_acts, s=100, 
                           cmap='viridis', alpha=0.7,
                           edgecolors='black', linewidth=1)
        
        # Formatting
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        ax.set_xlabel('Elbow Angle (degrees)')
        ax.set_ylabel('Elbow Torque (Nm)')
        ax.set_title('Asatryan & Feldman Unloading Response')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar for muscle activation
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Bicep Activation')
        
        # Clean up the plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return fig

def run_asatryan_feldman_experiment():
    """Main function to run the complete experiment."""
    
    print("🔬 Running Asatryan & Feldman Unloading Response Experiment")
    print("=" * 60)
    
    # Initialize experiment
    experiment = AsatryanFeldmanExperiment()
    
    # Run trial with weight perturbations
    print("Running weight perturbation trial...")
    data_log = experiment.run_weight_perturbation_trial(
        weights=[0.0, 0.2, 0.4, 0.6, 0.2, 0.0],  # kg
        weight_durations=5.0,  # seconds
        max_steps=300
    )
    
    # Analyze results
    print("Analyzing unloading responses...")
    steady_state_results = experiment.analyze_unloading_responses(data_log)
    
    # Plot results
    print("Creating Asatryan & Feldman plot...")
    fig = experiment.plot_asatryan_feldman_results(
        steady_state_results,
        save_path="asatryan_feldman_results.png"
    )
    
    print("✅ Experiment completed!")
    print(f"Found {len(steady_state_results)} weight conditions")
    print("Results saved to: asatryan_feldman_results.png")
    
    return data_log, steady_state_results

if __name__ == "__main__":
    data_log, results = run_asatryan_feldman_experiment()