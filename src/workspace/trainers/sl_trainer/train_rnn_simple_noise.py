import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pickle
import os
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt

from networks.rnn import RNNPolicy
from dataset import DemonstrationDataset

def train_with_noise_injection(policy, dataloader, optimizer, criterion, max_grad_norm=1.0, 
                              device='cpu', noise_std=0.015, noise_prob=0.4):
    """Simple training with noise injection - no spectral radius clamping."""
    
    policy.train()
    total_loss = 0.0
    num_batches = 0
    grad_norms = []
    noise_applications = 0
    
    for batch_obs, batch_actions, batch_masks in tqdm(dataloader, desc="Training with Noise"):
        batch_obs = batch_obs.to(device)
        batch_actions = batch_actions.to(device)
        batch_masks = batch_masks.to(device)
        
        # SIMPLE NOISE INJECTION
        apply_noise = torch.rand(1) < noise_prob
        if apply_noise:
            noise_applications += 1
            # Gaussian noise
            noise = torch.randn_like(batch_obs) * noise_std
            
            # Reduce noise on target encoding (dims 12-28) to preserve important info
            noise_mask = torch.ones_like(noise)
            noise_mask[:, :, 12:28] *= 0.3  # Less noise on target encoding
            noise = noise * noise_mask
            
            batch_obs = batch_obs + noise
        
        # Standard training step
        optimizer.zero_grad()
        hidden = policy.init_hidden(batch_obs.size(0), device=device)
        predicted_actions, _ = policy(batch_obs, hidden)
        
        # Loss computation
        loss = criterion(predicted_actions, batch_actions)
        if batch_masks is not None:
            loss = loss * batch_masks.unsqueeze(-1)
            loss = loss.sum() / batch_masks.sum()
        
        loss.backward()
        
        # Gradient clipping
        total_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=float('inf'))
        grad_norms.append(total_norm.item())
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Occasional logging
        if num_batches % 50 == 0:
            noise_status = f"(+noise)" if apply_noise else ""
            print(f"   Batch {num_batches}: Loss = {loss:.6f} {noise_status}")
    
    noise_rate = noise_applications / num_batches
    avg_grad_norm = sum(grad_norms) / len(grad_norms)
    
    print(f"   📈 Average gradient norm: {avg_grad_norm:.4f}")
    print(f"   🎲 Noise applied to {noise_applications}/{num_batches} batches ({noise_rate:.1%})")
    
    return total_loss / num_batches, grad_norms

def train_simple_robust_rnn(demonstrations_path="sac_demonstrations_50steps_successful_selective5.pkl",
                           save_dir="./rnn_simple_noise", 
                           selective_norm_path="selective_vec_normalize3.pkl",
                           num_epochs=200,  # Shorter for initial test
                           batch_size=16,
                           sequence_length=50,
                           learning_rate=1e-3,
                           hidden_size=64,
                           tau_mem=1.0,
                           recurrent_lr_factor=0.1,
                           # Simple noise parameters
                           noise_std=0.015,
                           noise_prob=0.4):
    """Simple robust training - just noise injection, no spectral radius clamping."""
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Copy normalization file
    if os.path.exists(selective_norm_path):
        dest_norm_path = os.path.join(save_dir, 'selective_vec_normalize.pkl')
        shutil.copy2(selective_norm_path, dest_norm_path)
        print(f"💾 Copied normalization to: {dest_norm_path}")
    
    # Load demonstrations
    print(f"Loading demonstrations from {demonstrations_path}...")
    with open(demonstrations_path, 'rb') as f:
        demonstrations = pickle.load(f)
    
    # Get dimensions
    obs_dim = demonstrations['observations'].shape[1]
    action_dim = demonstrations['actions'].shape[1]
    
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    print(f"\n🎲 NOISE CONFIGURATION:")
    print(f"   Noise std: {noise_std:.4f}")
    print(f"   Noise probability: {noise_prob:.1%}")
    print(f"   NO spectral radius clamping (simplified)")
    
    # Create dataset and dataloader
    dataset = DemonstrationDataset(demonstrations, sequence_length=sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    print(f"Created {len(dataset)} training sequences")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = RNNPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
        activation=nn.Sigmoid(),
        alpha=1.0/tau_mem
    ).to(device)
    
    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    # Simple optimizer - different learning rates for recurrent vs other params
    rec_params = []
    other_params = []
    
    for name, param in policy.named_parameters():
        if "W_h" in name:
            rec_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = optim.Adam([
        {"params": rec_params, "lr": learning_rate * recurrent_lr_factor},
        {"params": other_params, "lr": learning_rate}
    ])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.7)
    criterion = nn.MSELoss()
    
    # Training loop
    losses = []
    best_loss = float('inf')
    all_grad_norms = []
    
    print("\nStarting simple robust training...")
    
    for epoch in range(num_epochs):
        avg_loss, epoch_grad_norms = train_with_noise_injection(
            policy=policy,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            max_grad_norm=1.0,
            device=device,
            noise_std=noise_std,
            noise_prob=noise_prob
        )
        
        scheduler.step(avg_loss)
        losses.append(avg_loss)
        all_grad_norms.extend(epoch_grad_norms)
        
        print(f"Epoch {epoch+1:3d}: Loss = {avg_loss:.6f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'config': {
                    'obs_dim': obs_dim,
                    'action_dim': action_dim,
                    'hidden_size': hidden_size,
                    'tau_mem': tau_mem,
                    'sequence_length': sequence_length,
                    'normalization_file': 'selective_vec_normalize.pkl',
                    'training_data': demonstrations_path,
                    'noise_training': True,
                    'noise_std': noise_std,
                    'noise_prob': noise_prob,
                    'spectral_clamping': False  # Explicitly mark as disabled
                }
            }, os.path.join(save_dir, 'best_rnn_simple_noise.pth'))
            print(f"   💾 New best model saved! Loss: {avg_loss:.6f}")
    
    # Simple plotting
    plt.figure(figsize=(12, 4))
    
    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss (with Noise Injection)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.grid(True)
    
    # Gradient norms
    plt.subplot(1, 2, 2)
    plt.hist(all_grad_norms, bins=50, alpha=0.7)
    plt.title('Gradient Norm Distribution')
    plt.xlabel('Gradient Norm')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'simple_noise_training.png'), dpi=150)
    plt.close()
    
    print("\n" + "="*50)
    print("🎲 SIMPLE NOISE TRAINING COMPLETED!")
    print("="*50)
    print(f"Best loss: {best_loss:.6f}")
    print(f"Models saved in: {save_dir}")
    print(f"Noise std used: {noise_std:.4f}")
    print(f"Noise probability: {noise_prob:.1%}")
    print("\n💡 Next steps:")
    print("   1. Test this model vs your baseline RNN")
    print("   2. If noise helps, try different noise levels")
    print("   3. If noise doesn't help enough, consider DAgger")
    
    return policy

if __name__ == "__main__":
    # Simple test with just noise injection
    train_simple_robust_rnn(
        demonstrations_path="sac_demonstrations_50steps_successful_selective5.pkl",
        save_dir="./rnn_simple_noise_test",
        selective_norm_path="selective_vec_normalize3.pkl",
        num_epochs=200,  # Shorter for quick test
        batch_size=16,
        sequence_length=50,
        learning_rate=1e-3,
        recurrent_lr_factor=0.1,
        hidden_size=64,
        tau_mem=1.0,
        # Just noise parameters
        noise_std=0.015,
        noise_prob=0.4
    )