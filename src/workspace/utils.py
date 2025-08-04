"""Simple modular utilities for component compatibility."""

import torch
import numpy as np

def encode_numpy(encoder, observation: np.ndarray) -> np.ndarray:
    """Helper function to encode numpy arrays using PyTorch encoders."""
    with torch.no_grad():
        # Convert to tensor
        obs_tensor = torch.FloatTensor(observation)
        
        # Add batch dimension if needed
        single_sample = obs_tensor.dim() == 1
        if single_sample:
            obs_tensor = obs_tensor.unsqueeze(0)
        
        # Encode
        encoded_tensor = encoder(obs_tensor)
        
        # Convert back to numpy
        encoded = encoded_tensor.numpy()
        
        # Remove batch dimension if it was added
        if single_sample:
            encoded = encoded.squeeze(0)
            
        return encoded

def validate_compatibility(encoder, policy):
    """Validate encoder and policy can work together."""
    if not hasattr(encoder, 'output_dim'):
        raise AttributeError(f"{type(encoder).__name__} missing 'output_dim' property")
    if not hasattr(policy, 'input_dim'):
        raise AttributeError(f"{type(policy).__name__} missing 'input_dim' property")
    
    if encoder.output_dim != policy.input_dim:
        raise ValueError(f"Dimension mismatch: {type(encoder).__name__}({encoder.output_dim}) → {type(policy).__name__}({policy.input_dim})")
    
    print(f"✅ Compatible: {type(encoder).__name__}({encoder.output_dim}) → {type(policy).__name__}({policy.input_dim})")

def print_component_info(encoder, policy=None, trainer_name=None):
    """Print information about the components."""
    print(f"\n🔧 COMPONENT SETUP")
    print(f"   Encoder: {type(encoder).__name__}")
    print(f"   Input dim: {getattr(encoder, 'input_dim', 'unknown')}")
    print(f"   Output dim: {encoder.output_dim}")
    
    if policy:
        print(f"   Policy: {type(policy).__name__}")
        print(f"   Policy input: {policy.input_dim}")
        print(f"   Policy output: {getattr(policy, 'output_dim', getattr(policy, 'action_dim', 'unknown'))}")
    
    if trainer_name:
        print(f"   Trainer: {trainer_name}")
    print()