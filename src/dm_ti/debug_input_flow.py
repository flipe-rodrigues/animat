import torch
import numpy as np
from networks import ModalitySpecificEncoder
from shimmy_wrapper import create_env

def debug_input_flow(num_samples=5):
    """Trace the entire input flow through normalization and encoding"""
    
    # Create environment and get a sample observation
    env = create_env()
    observations = []
    
    # Collect sample observations
    obs, _ = env.reset()
    observations.append(obs)
    
    # Get a few more samples
    for _ in range(num_samples-1):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        observations.append(obs)
    
    # Create encoder
    encoder = ModalitySpecificEncoder(target_size=40)
    
    print("\n===== INPUT FLOW DEBUG REPORT =====\n")
    
    # Analyze each observation
    for i, obs in enumerate(observations):
        print(f"\nSample {i+1}/{len(observations)}:")
        
        # Original observation
        print(f"  Raw Observation Shape: {obs.shape}")
        print(f"  Muscle Data (first 5): {obs[:5]}")
        print(f"  Target Pos (X,Y,Z): {obs[12:15]}")
        
        # Convert to tensor for encoder
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        
        # Get encoder's preferred values for XY coordinates
        print("\n  Population Encoder Config:")
        x_pref = encoder.target_encoder.preferred_values[0].cpu().numpy()
        y_pref = encoder.target_encoder.preferred_values[1].cpu().numpy()
        print(f"    X preferred values range: {x_pref.min():.4f} to {x_pref.max():.4f}")
        print(f"    Y preferred values range: {y_pref.min():.4f} to {y_pref.max():.4f}")
        print(f"    Gaussian sigma: {encoder.target_encoder.sigma.cpu().numpy()}")
        
        # Encode and analyze
        encoded = encoder(obs_tensor)
        
        print("\n  Encoded Output:")
        print(f"    Shape: {encoded.shape}")  # Should be [1, 93] (12+80+1)
        print(f"    Muscle part (first 5): {encoded[0, :5].detach().cpu().numpy()}")
        
        # Analyze population encoding for X
        x_pop = encoded[0, 12:12+40].detach().cpu().numpy()
        print(f"    X encoding (max activation): {x_pop.max():.4f} at index {np.argmax(x_pop)}")
        
        # Analyze population encoding for Y
        y_pop = encoded[0, 12+40:12+80].detach().cpu().numpy()
        print(f"    Y encoding (max activation): {y_pop.max():.4f} at index {np.argmax(y_pop)}")
        
        # Z value
        z_value = encoded[0, -1].item()
        print(f"    Z value (unchanged): {z_value:.4f}")
        
    print("\n===== END DEBUG REPORT =====\n")
    env.close()

if __name__ == "__main__":
    debug_input_flow()