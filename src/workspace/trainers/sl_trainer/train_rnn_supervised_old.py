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
import sys
from pathlib import Path

workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

from wrappers.sl_wrappers.dataset import DemonstrationDataset
from networks.rnn import RNNPolicy  # Change this import


def clamp_spectral_radius_fn(weight_hh, rho_target=0.9):  # Renamed function
    """Clamp spectral radius to target value."""
    with torch.no_grad():
        # Get current spectral radius
        eigenvalues = torch.linalg.eigvals(weight_hh)
        current_rho = torch.max(torch.abs(eigenvalues)).item()
        
        if current_rho > rho_target:
            # Scale down the weights to achieve target spectral radius
            scale_factor = rho_target / current_rho
            weight_hh.mul_(scale_factor)
            return current_rho, rho_target
        return current_rho, current_rho

# Modified train_with_full_bptt function with spectral radius clamping
def train_with_full_bptt(policy, dataloader, optimizer, criterion, max_grad_norm=1.0, device='cpu', 
                        enable_spectral_clamping=True, rho_target=0.9):  # Renamed parameter
    """Train with full BPTT with optional spectral radius clamping."""
    
    policy.train()
    total_loss = 0.0
    num_batches = 0
    grad_norms = []
    spectral_radius_info = []
    
    for batch_obs, batch_actions, batch_masks in tqdm(dataloader, desc="Training (Full BPTT)"):
        batch_obs = batch_obs.to(device)
        batch_actions = batch_actions.to(device)
        batch_masks = batch_masks.to(device)
        
        optimizer.zero_grad()
        hidden = policy.init_hidden(batch_obs.size(0), device=device)
        predicted_actions, _ = policy(batch_obs, hidden)
        
        # Apply mask to loss computation
        loss = criterion(predicted_actions, batch_actions)
        if batch_masks is not None:
            loss = loss * batch_masks.unsqueeze(-1)
            loss = loss.sum() / batch_masks.sum()
        
        loss.backward()
        
        # Track gradient norm before clipping
        total_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=float('inf'))
        grad_norms.append(total_norm.item())
        
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()
        
        # SPECTRAL RADIUS CLAMPING after optimizer step
        if enable_spectral_clamping:  # Fixed parameter name
            current_rho, final_rho = clamp_spectral_radius_fn(policy.W_h, rho_target)  # Fixed function call
            spectral_radius_info.append((current_rho, final_rho))
            
            # Log clamping info occasionally
            if num_batches % 50 == 0 and current_rho != final_rho:
                print(f"   🔧 Batch {num_batches}: Spectral radius clamped from {current_rho:.4f} to {final_rho:.4f}")
        
        total_loss += loss.item()
        num_batches += 1
        
        # Log gradient norm every 20 batches
        if num_batches % 20 == 0:
            print(f"   ⚡ Batch {num_batches}: Loss = {loss:.6f}, Grad norm = {total_norm:.4f}")
    
    avg_grad_norm = sum(grad_norms) / len(grad_norms)
    max_grad_norm_val = max(grad_norms)
    print(f"   📈 Average gradient norm: {avg_grad_norm:.4f}, Max: {max_grad_norm_val:.4f}")
    
    # Report spectral radius clamping statistics
    if enable_spectral_clamping and spectral_radius_info:  # Fixed parameter name
        clamped_count = sum(1 for curr, final in spectral_radius_info if curr != final)
        avg_initial_rho = np.mean([curr for curr, final in spectral_radius_info])
        avg_final_rho = np.mean([final for curr, final in spectral_radius_info])
        print(f"   🔧 Spectral radius: {clamped_count}/{len(spectral_radius_info)} steps clamped")
        print(f"   📊 Average ρ before/after: {avg_initial_rho:.4f} → {avg_final_rho:.4f}")
    
    return total_loss / num_batches, grad_norms


def calculate_spectral_radius(weights):
    """Calculate the spectral radius of a weight matrix."""
    eigenvalues = np.linalg.eigvals(weights)
    return np.max(np.abs(eigenvalues))

def save_weight_checkpoint(policy, epoch, save_dir):
    """Save RNN weights for analysis - updated for NumpyStyleRNNPolicy."""
    os.makedirs(os.path.join(save_dir, 'weight_checkpoints'), exist_ok=True)
    
    # Extract recurrent weights from NumpyStyleRNNPolicy
    weight_hh = policy.W_h.detach().cpu().numpy()  # Changed from policy.rnn.cells[0].weight_hh
    
    # Save weights
    checkpoint_path = os.path.join(save_dir, 'weight_checkpoints', f'weight_hh_epoch_{epoch}.npy')
    np.save(checkpoint_path, weight_hh)
    
    # Calculate and print spectral radius
    spectral_radius = calculate_spectral_radius(weight_hh)
    print(f"   📊 W_h spectral radius at epoch {epoch}: {spectral_radius:.4f}")
    
    return spectral_radius


def train_supervised_rnn(demonstrations_path="sac_demonstrations_50steps_successful_selective.pkl",  # Updated default
                        save_dir="./rnn_supervised_models", 
                        selective_norm_path="selective_vec_normalize.pkl",  # Add this parameter
                        use_tbptt=False,
                        chunk_size=32,
                        accumulate_steps=3,
                        num_epochs=100, 
                        batch_size=8,
                        sequence_length=50,  # Updated to match new data
                        learning_rate=5e-4,
                        hidden_size=128,
                        tau_mem=1.0,
                        recurrent_lr_factor=0.1,
                        clamp_spectral_radius=True,  # NEW: Enable spectral radius clamping
                        rho_target=0.9):  # NEW: Target spectral radius
    """
    Main training function with selective normalization support and spectral radius clamping.
    """
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Copy selective normalization to save directory for consistency
    if os.path.exists(selective_norm_path):
        dest_norm_path = os.path.join(save_dir, 'selective_vec_normalize.pkl')
        shutil.copy2(selective_norm_path, dest_norm_path)
        print(f"💾 Copied selective normalization to: {dest_norm_path}")
    else:
        print(f"⚠️  Warning: {selective_norm_path} not found!")
    
    # Load demonstrations
    print(f"Loading demonstrations from {demonstrations_path}...")
    with open(demonstrations_path, 'rb') as f:
        demonstrations = pickle.load(f)
    
    print(f"Loaded {len(demonstrations['observations'])} demonstrations")
    
    # Check metadata if available
    if 'metadata' in demonstrations:
        metadata = demonstrations['metadata']
        print(f"Demonstration metadata:")
        print(f"  Collection type: {metadata.get('collection_type', 'unknown')}")
        print(f"  Normalization type: {metadata.get('normalization_type', 'unknown')}")
        print(f"  Episodes collected: {metadata.get('num_episodes_collected', 'unknown')}")
        print(f"  Success rate: {metadata.get('success_rate', 'unknown'):.3f}")
    
    # Get dimensions
    obs_dim = demonstrations['observations'].shape[1]
    action_dim = demonstrations['actions'].shape[1]
    
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    print(f"Using Full BPTT with sequence length: {sequence_length}")
    
    # NEW: Print spectral radius clamping info
    if clamp_spectral_radius:
        print(f"🔧 SPECTRAL RADIUS CLAMPING ENABLED: target ρ = {rho_target}")
    else:
        print(f"⚠️  SPECTRAL RADIUS CLAMPING DISABLED")
    
    # Create dataset and dataloader
    dataset = DemonstrationDataset(demonstrations, sequence_length=sequence_length)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4
    )
    
    print(f"Created {len(dataset)} training sequences")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = RNNPolicy(  # Changed from LeakyRNNPolicy
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
        activation=nn.Sigmoid(),  # Add activation parameter
        alpha=1.0/tau_mem  # Convert tau_mem to alpha
    ).to(device)
    
    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    # Create parameter groups with different learning rates for NumpyStyleRNNPolicy
    rec_params = []
    other_params = []
    
    for name, param in policy.named_parameters():
        if "W_h" in name:  # Changed from "rnn.cells.0.weight_hh"
            rec_params.append(param)
            print(f"   🎯 Recurrent weight: {name} (reduced LR)")
        else:
            other_params.append(param)
    
    print(f"   📊 Recurrent params: {len(rec_params)}, Other params: {len(other_params)}")
    
    # Optimizer with different learning rates
    optimizer = optim.Adam([
        {"params": rec_params, "lr": learning_rate * recurrent_lr_factor},  # 10x smaller for W_hh
        {"params": other_params, "lr": learning_rate}
    ], weight_decay=1e-5)
    
    print(f"   🔧 Recurrent LR: {learning_rate * recurrent_lr_factor:.6f}")
    print(f"   🔧 Other LR: {learning_rate:.6f}")
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    criterion = nn.MSELoss()
    
    # Training loop
    losses = []
    best_loss = float('inf')
    all_grad_norms = []
    spectral_radii = []
    epochs_to_check = [0, 10, 20, 50, 75, 100, 150, 200, 250]  # Updated for longer training
    
    # Save initial weights
    initial_spectral_radius = save_weight_checkpoint(policy, 0, save_dir)
    spectral_radii.append((0, initial_spectral_radius))
    
    print("Starting supervised training with spectral radius clamping...")
    
    for epoch in range(num_epochs):
        avg_loss, epoch_grad_norms = train_with_full_bptt(
            policy=policy,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            max_grad_norm=1.0,
            device=device,
            enable_spectral_clamping=clamp_spectral_radius,  # Fixed parameter name
            rho_target=rho_target  # NEW: Pass target spectral radius
        )
        
        scheduler.step(avg_loss)
        losses.append(avg_loss)
        all_grad_norms.extend(epoch_grad_norms)
        
        print(f"Epoch {epoch+1:3d}: Loss = {avg_loss:.6f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
        
        # Check spectral radius at specific epochs
        if (epoch + 1) in epochs_to_check:
            sr = save_weight_checkpoint(policy, epoch + 1, save_dir)
            spectral_radii.append((epoch + 1, sr))
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            model_name = 'best_rnn_full.pth'
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
                    'normalization_file': 'selective_vec_normalize.pkl',  # Reference to norm file
                    'normalization_type': 'SelectiveVecNormalize_first_12_dims',
                    'training_data': demonstrations_path,
                    'spectral_radius_clamping': clamp_spectral_radius,  # NEW: Save clamping info
                    'rho_target': rho_target if clamp_spectral_radius else None  # NEW: Save target spectral radius
                }
            }, os.path.join(save_dir, model_name))
            print(f"   💾 New best model saved! Loss: {avg_loss:.6f}")
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_name = f'rnn_checkpoint_epoch_{epoch+1}.pth'
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
                    'normalization_type': 'SelectiveVecNormalize_first_12_dims',
                    'training_data': demonstrations_path,
                    'spectral_radius_clamping': clamp_spectral_radius,
                    'rho_target': rho_target if clamp_spectral_radius else None
                }
            }, os.path.join(save_dir, checkpoint_name))
    
    # Plot training curve and metrics
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Loss curve
    plt.subplot(2, 2, 1)
    plt.plot(losses)
    plt.title('RNN Supervised Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.grid(True)
    
    # Plot 2: Gradient norms
    plt.subplot(2, 2, 2)
    plt.plot(all_grad_norms)
    plt.title('Gradient Norms During Training')
    plt.xlabel('Update Step')
    plt.ylabel('Gradient Norm')
    plt.yscale('log')
    plt.grid(True)
    
    # Plot 3: Spectral radius evolution
    plt.subplot(2, 2, 3)
    epochs, radii = zip(*spectral_radii)
    plt.plot(epochs, radii, 'o-', linewidth=2)
    plt.title('Spectral Radius Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Spectral Radius')
    if clamp_spectral_radius:
        plt.axhline(y=rho_target, color='red', linestyle='--', alpha=0.7, label=f'Target ρ = {rho_target}')
        plt.legend()
    plt.grid(True)
    
    # Plot 4: Gradient norm distribution
    plt.subplot(2, 2, 4)
    plt.hist(all_grad_norms, bins=50)
    plt.title('Gradient Norm Distribution')
    plt.xlabel('Gradient Norm')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics_spectral_clamped.png'))
    plt.close()
    
    # Print summary
    print("\n📊 SPECTRAL RADIUS EVOLUTION:")
    print("=" * 40)
    for epoch, radius in spectral_radii:
        status = "✅ CLAMPED" if clamp_spectral_radius and radius <= rho_target + 0.01 else ""
        print(f"   Epoch {epoch:3d}: {radius:.4f} {status}")
    
    print(f"\n🎉 TRAINING COMPLETED!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Models saved in: {save_dir}")
    if clamp_spectral_radius:
        print(f"🔧 Spectral radius clamping: ENABLED (target ρ = {rho_target})")
    print(f"🔧 IMPORTANT: Use 'selective_vec_normalize.pkl' for evaluation/inference")
    
    return policy


if __name__ == "__main__":
    # Update paths to match your actual files
    train_supervised_rnn(
        demonstrations_path="sac_demonstrations_50steps_successful_selective5.pkl",  # ✅ This matches your file
        save_dir="./models/rnn_test_64hidden_spectral_clamped4",
        selective_norm_path="models/selective_vec_normalize3.pkl",  # Add models/ prefix
        num_epochs=400,
        batch_size=16,
        sequence_length=50,
        learning_rate=1e-3,
        recurrent_lr_factor=0.1,
        hidden_size=64,
        tau_mem=1.0,
        clamp_spectral_radius=True,
        rho_target=0.95
    )
