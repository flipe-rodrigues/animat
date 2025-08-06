"""Test script for supervised learning with different encoders and networks."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

from networks.rnn import RNNPolicy
from encoders.encoders import IdentityEncoder, ModalitySpecificEncoder, GridEncoder
from envs.dm_env import make_arm_env

class MLPPolicy(nn.Module):
    """Simple MLP for comparison with RNN."""
    
    def __init__(self, input_dim, action_dim, hidden_sizes=[64, 64]):
        super().__init__()
        
        layers = []
        prev_size = input_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, action_dim))
        layers.append(nn.Sigmoid())  # Muscle activations in [0,1]
        
        self.network = nn.Sequential(*layers)
        self._input_dim = input_dim
        self._action_dim = action_dim
    
    @property
    def input_dim(self):
        return self._input_dim
    
    @property
    def output_dim(self):
        return self._action_dim
    
    def forward(self, x):
        return self.network(x)

def generate_demonstration_data(num_episodes=50, max_steps=100):
    """Generate demonstration data by random policy."""
    print("🎲 Generating demonstration data...")
    
    observations = []
    actions = []
    
    for episode in range(num_episodes):
        env = make_arm_env(random_seed=episode)
        timestep = env.reset()
        
        episode_obs = []
        episode_actions = []
        
        for step in range(max_steps):
            # Extract observation
            muscle_data = timestep.observation['muscle_sensors']
            target_pos = timestep.observation['target_position'].flatten()
            obs = np.concatenate([muscle_data, target_pos])
            
            # Random action (in valid range)
            action = np.random.uniform(0, 1, size=4)
            
            episode_obs.append(obs)
            episode_actions.append(action)
            
            timestep = env.step(action)
            
            if timestep.last():
                break
        
        observations.extend(episode_obs)
        actions.extend(episode_actions)
        env.close()
    
    print(f"   Generated {len(observations)} samples from {num_episodes} episodes")
    return np.array(observations), np.array(actions)

def test_supervised_learning():
    """Test supervised learning with different encoder-network combinations."""
    print("\n📚 TESTING SUPERVISED LEARNING")
    print("=" * 60)
    
    # Generate demonstration data
    X_raw, y = generate_demonstration_data(num_episodes=20, max_steps=50)
    
    # Test configurations
    configs = [
        ("MLP_Identity", MLPPolicy, IdentityEncoder(obs_dim=15)),
        ("MLP_Grid3", MLPPolicy, ModalitySpecificEncoder(grid_size=3, raw_obs_dim=15)),
        ("MLP_Grid5", MLPPolicy, ModalitySpecificEncoder(grid_size=5, raw_obs_dim=15)),
        ("RNN_Identity", RNNPolicy, IdentityEncoder(obs_dim=15)),
        ("RNN_Grid3", RNNPolicy, ModalitySpecificEncoder(grid_size=3, raw_obs_dim=15)),
        ("RNN_Grid5", RNNPolicy, ModalitySpecificEncoder(grid_size=5, raw_obs_dim=15)),
    ]
    
    results = {}
    
    for config_name, network_class, encoder in configs:
        try:
            print(f"\n🧠 Testing {config_name}...")
            
            # Encode observations
            X_encoded = []
            for obs in X_raw:
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs)
                    encoded = encoder(obs_tensor).numpy()
                    X_encoded.append(encoded)
            X_encoded = np.array(X_encoded)
            
            print(f"   📊 Data: {X_raw.shape} → {X_encoded.shape}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_encoded, y, test_size=0.2, random_state=42
            )
            
            # Create datasets
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train), 
                torch.FloatTensor(y_train)
            )
            test_dataset = TensorDataset(
                torch.FloatTensor(X_test), 
                torch.FloatTensor(y_test)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32)
            
            # Create network
            if network_class == RNNPolicy:
                network = RNNPolicy(
                    input_dim=encoder.output_dim,
                    action_dim=4,
                    hidden_size=32,
                    alpha=0.1
                )
            else:  # MLPPolicy
                network = MLPPolicy(
                    input_dim=encoder.output_dim,
                    action_dim=4,
                    hidden_sizes=[64, 32]
                )
            
            # Training setup
            optimizer = optim.Adam(network.parameters(), lr=1e-3)
            criterion = nn.MSELoss()
            
            # Training loop
            network.train()
            train_losses = []
            
            for epoch in range(10):  # Short training for testing
                epoch_loss = 0
                for batch_obs, batch_actions in train_loader:
                    optimizer.zero_grad()
                    
                    if isinstance(network, RNNPolicy):
                        pred_actions, _ = network(batch_obs)
                    else:
                        pred_actions = network(batch_obs)
                    
                    loss = criterion(pred_actions, batch_actions)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(train_loader)
                train_losses.append(avg_loss)
                
                if epoch % 5 == 0:
                    print(f"      Epoch {epoch}: Loss = {avg_loss:.6f}")
            
            # Evaluation
            network.eval()
            test_loss = 0
            with torch.no_grad():
                for batch_obs, batch_actions in test_loader:
                    if isinstance(network, RNNPolicy):
                        pred_actions, _ = network(batch_obs)
                    else:
                        pred_actions = network(batch_obs)
                    
                    loss = criterion(pred_actions, batch_actions)
                    test_loss += loss.item()
            
            avg_test_loss = test_loss / len(test_loader)
            final_train_loss = train_losses[-1]
            
            print(f"   ✅ Training completed")
            print(f"      Final train loss: {final_train_loss:.6f}")
            print(f"      Test loss: {avg_test_loss:.6f}")
            
            results[config_name] = {
                'encoder_dim': encoder.output_dim,
                'train_loss': final_train_loss,
                'test_loss': avg_test_loss,
                'network_params': sum(p.numel() for p in network.parameters()),
                'status': 'success'
            }
            
        except Exception as e:
            print(f"   ❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            results[config_name] = {'status': 'failed', 'error': str(e)}
    
    return results

def test_encoder_visualization():
    """Test encoder outputs with visualization."""
    print("\n🎨 TESTING ENCODER VISUALIZATION")
    print("=" * 60)
    
    # Generate test observations
    test_obs = []
    for i in range(10):
        muscle_data = np.random.uniform(0, 1, 12)
        target_pos = np.random.uniform([-0.5, -0.5, 0.1], [0.5, 0.5, 0.3], 3)
        obs = np.concatenate([muscle_data, target_pos])
        test_obs.append(obs)
    test_obs = np.array(test_obs)
    
    encoders = [
        ("Identity", IdentityEncoder(obs_dim=15)),
        ("Grid_3x3", ModalitySpecificEncoder(grid_size=3, raw_obs_dim=15)),
        ("Grid_5x5", ModalitySpecificEncoder(grid_size=5, raw_obs_dim=15)),
    ]
    
    fig, axes = plt.subplots(1, len(encoders), figsize=(15, 4))
    
    for idx, (name, encoder) in enumerate(encoders):
        print(f"   🔍 Testing {name} encoder...")
        
        encoded_outputs = []
        for obs in test_obs:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs)
                encoded = encoder(obs_tensor).numpy()
                encoded_outputs.append(encoded)
        
        encoded_outputs = np.array(encoded_outputs)
        print(f"      Input: {test_obs.shape} → Output: {encoded_outputs.shape}")
        
        # Visualize encoded outputs
        im = axes[idx].imshow(encoded_outputs.T, aspect='auto', cmap='viridis')
        axes[idx].set_title(f'{name}\n{encoded_outputs.shape[1]}D output')
        axes[idx].set_xlabel('Sample')
        axes[idx].set_ylabel('Feature')
        plt.colorbar(im, ax=axes[idx])
    
    plt.tight_layout()
    plt.savefig('encoder_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("   💾 Saved visualization to 'encoder_comparison.png'")

if __name__ == "__main__":
    print("🚀 SUPERVISED LEARNING COMPREHENSIVE TEST")
    print("=" * 80)
    
    # Test supervised learning
    sl_results = test_supervised_learning()
    
    # Test encoder visualization
    test_encoder_visualization()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 SUPERVISED LEARNING RESULTS")
    print("=" * 80)
    
    for name, result in sl_results.items():
        if result['status'] == 'success':
            print(f"✅ {name}:")
            print(f"   Encoder dim: {result['encoder_dim']}")
            print(f"   Parameters: {result['network_params']:,}")
            print(f"   Train loss: {result['train_loss']:.6f}")
            print(f"   Test loss: {result['test_loss']:.6f}")
        else:
            print(f"❌ {name}: {result['error']}")
    
    # Overall assessment
    passed_tests = sum(1 for r in sl_results.values() if r['status'] == 'success')
    total_tests = len(sl_results)
    
    print(f"\n🎉 OVERALL: {passed_tests}/{total_tests} SL tests passed")