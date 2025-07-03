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

from policy import NumpyStyleRNNPolicy
from dataset import DemonstrationDataset

def clamp_spectral_radius_fn(weight_hh, rho_target=0.9):
    """Clamp spectral radius to target value."""
    with torch.no_grad():
        eigenvalues = torch.linalg.eigvals(weight_hh)
        current_rho = torch.max(torch.abs(eigenvalues)).item()
        
        if current_rho > rho_target:
            scale_factor = rho_target / current_rho
            weight_hh.mul_(scale_factor)
            return current_rho, rho_target
        return current_rho, current_rho

def train_with_full_bptt_robust(policy, dataloader, optimizer, criterion, max_grad_norm=1.0, 
                               device='cpu', enable_spectral_clamping=True, rho_target=0.9,
                               noise_std=0.02, noise_prob=0.5, noise_type='gaussian'):
    """
    Enhanced training with observation noise for robustness against distributional shift.
    
    Args:
        noise_std: Standard deviation of noise to add
        noise_prob: Probability of adding noise to a batch
        noise_type: 'gaussian', 'uniform', or 'mixed'
    """
    
    policy.train()
    total_loss = 0.0
    num_batches = 0
    grad_norms = []
    spectral_radius_info = []
    noise_applications = 0
    
    for batch_obs, batch_actions, batch_masks in tqdm(dataloader, desc="Training (Robust BPTT)"):
        batch_obs = batch_obs.to(device)
        batch_actions = batch_actions.to(device)
        batch_masks = batch_masks.to(device)
        
        # NOISE INJECTION for robustness
        apply_noise = torch.rand(1) < noise_prob
        if apply_noise:
            noise_applications += 1
            
            if noise_type == 'gaussian':
                # Gaussian noise - most common
                noise = torch.randn_like(batch_obs) * noise_std
            elif noise_type == 'uniform':
                # Uniform noise - different distribution
                noise = (torch.rand_like(batch_obs) - 0.5) * 2 * noise_std * 1.732  # Scale for same std
            elif noise_type == 'mixed':
                # Mix of both - more diverse
                if torch.rand(1) < 0.5:
                    noise = torch.randn_like(batch_obs) * noise_std
                else:
                    noise = (torch.rand_like(batch_obs) - 0.5) * 2 * noise_std * 1.732
            
            # Apply noise only to relevant dimensions (avoid corrupting target encoding too much)
            # Muscle sensors (0-12): full noise
            # Target encoding (12-28): reduced noise (these are important!)
            # Other dims (28+): moderate noise
            noise_mask = torch.ones_like(noise)
            noise_mask[:, :, 12:28] *= 0.3  # Reduce noise on target encoding
            noise = noise * noise_mask
            
            batch_obs = batch_obs + noise
        
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
        if enable_spectral_clamping:
            current_rho, final_rho = clamp_spectral_radius_fn(policy.W_h, rho_target)
            spectral_radius_info.append((current_rho, final_rho))
            
            # Log clamping info occasionally
            if num_batches % 100 == 0 and current_rho != final_rho:
                print(f"   🔧 Batch {num_batches}: Spectral radius clamped from {current_rho:.4f} to {final_rho:.4f}")
        
        total_loss += loss.item()
        num_batches += 1
        
        # Log progress every 50 batches
        if num_batches % 50 == 0:
            noise_status = f"(+noise)" if apply_noise else ""
            print(f"   ⚡ Batch {num_batches}: Loss = {loss:.6f}, Grad norm = {total_norm:.4f} {noise_status}")
    
    # Calculate statistics
    avg_grad_norm = sum(grad_norms) / len(grad_norms)
    max_grad_norm_val = max(grad_norms)
    noise_rate = noise_applications / num_batches
    
    print(f"   📈 Average gradient norm: {avg_grad_norm:.4f}, Max: {max_grad_norm_val:.4f}")
    print(f"   🎲 Noise applied to {noise_applications}/{num_batches} batches ({noise_rate:.1%})")
    
    # Report spectral radius clamping statistics
    if enable_spectral_clamping and spectral_radius_info:
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
    """Save RNN weights for analysis."""
    os.makedirs(os.path.join(save_dir, 'weight_checkpoints'), exist_ok=True)
    
    weight_hh = policy.W_h.detach().cpu().numpy()
    
    checkpoint_path = os.path.join(save_dir, 'weight_checkpoints', f'weight_hh_epoch_{epoch}.npy')
    np.save(checkpoint_path, weight_hh)
    
    spectral_radius = calculate_spectral_radius(weight_hh)
    print(f"   📊 W_h spectral radius at epoch {epoch}: {spectral_radius:.4f}")
    
    return spectral_radius

def train_robust_rnn(demonstrations_path="sac_demonstrations_50steps_successful_selective5.pkl",
                    save_dir="./rnn_robust_models", 
                    selective_norm_path="selective_vec_normalize3.pkl",
                    num_epochs=400, 
                    batch_size=16,
                    sequence_length=50,
                    learning_rate=1e-3,
                    hidden_size=64,
                    tau_mem=1.0,
                    recurrent_lr_factor=0.1,
                    clamp_spectral_radius=True,
                    rho_target=0.95,
                    # NEW: Robustness parameters
                    noise_std=0.015,
                    noise_prob=0.4,
                    noise_type='gaussian',
                    progressive_noise=True):
    """
    Train RNN with noise injection for robustness against distributional shift.
    
    Args:
        noise_std: Standard deviation of observation noise
        noise_prob: Probability of applying noise per batch
        noise_type: 'gaussian', 'uniform', or 'mixed'
        progressive_noise: Start with less noise, increase gradually
    """
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Copy selective normalization to save directory
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
    
    # Get dimensions
    obs_dim = demonstrations['observations'].shape[1]
    action_dim = demonstrations['actions'].shape[1]
    
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    print(f"Using Robust Full BPTT with sequence length: {sequence_length}")
    
    # Print robustness configuration
    print(f"\n🛡️  ROBUSTNESS CONFIGURATION:")
    print(f"   Noise std: {noise_std:.4f}")
    print(f"   Noise probability: {noise_prob:.1%}")
    print(f"   Noise type: {noise_type}")
    print(f"   Progressive noise: {progressive_noise}")
    
    if clamp_spectral_radius:
        print(f"🔧 SPECTRAL RADIUS CLAMPING ENABLED: target ρ = {rho_target}")
    
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
    policy = NumpyStyleRNNPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
        activation=nn.Sigmoid(),
        alpha=1.0/tau_mem
    ).to(device)
    
    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    # Create parameter groups with different learning rates
    rec_params = []
    other_params = []
    
    for name, param in policy.named_parameters():
        if "W_h" in name:
            rec_params.append(param)
            print(f"   🎯 Recurrent weight: {name} (reduced LR)")
        else:
            other_params.append(param)
    
    # Optimizer with different learning rates
    optimizer = optim.Adam([
        {"params": rec_params, "lr": learning_rate * recurrent_lr_factor},
        {"params": other_params, "lr": learning_rate}
    ], weight_decay=1e-5)
    
    print(f"   🔧 Recurrent LR: {learning_rate * recurrent_lr_factor:.6f}")
    print(f"   🔧 Other LR: {learning_rate:.6f}")
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    criterion = nn.MSELoss()
    
    # Training loop
    losses = []
    best_loss = float('inf')
    all_grad_norms = []
    spectral_radii = []
    noise_schedule = []
    epochs_to_check = [0, 10, 20, 50, 75, 100, 150, 200, 250, 300, 350, 400]
    
    # Save initial weights
    initial_spectral_radius = save_weight_checkpoint(policy, 0, save_dir)
    spectral_radii.append((0, initial_spectral_radius))
    
    print("\nStarting robust supervised training...")
    
    for epoch in range(num_epochs):
        # Progressive noise schedule
        if progressive_noise:
            # Start with 50% of target noise, gradually increase
            progress = min(epoch / (num_epochs * 0.3), 1.0)  # Reach full noise at 30% of training
            current_noise_std = noise_std * (0.5 + 0.5 * progress)
            current_noise_prob = noise_prob * (0.3 + 0.7 * progress)
        else:
            current_noise_std = noise_std
            current_noise_prob = noise_prob
        
        noise_schedule.append((current_noise_std, current_noise_prob))
        
        avg_loss, epoch_grad_norms = train_with_full_bptt_robust(
            policy=policy,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            max_grad_norm=1.0,
            device=device,
            enable_spectral_clamping=clamp_spectral_radius,
            rho_target=rho_target,
            noise_std=current_noise_std,
            noise_prob=current_noise_prob,
            noise_type=noise_type
        )
        
        scheduler.step(avg_loss)
        losses.append(avg_loss)
        all_grad_norms.extend(epoch_grad_norms)
        
        # Enhanced logging with noise info
        if progressive_noise:
            print(f"Epoch {epoch+1:3d}: Loss = {avg_loss:.6f}, LR = {optimizer.param_groups[0]['lr']:.6f}, "
                  f"Noise: σ={current_noise_std:.4f}, p={current_noise_prob:.2f}")
        else:
            print(f"Epoch {epoch+1:3d}: Loss = {avg_loss:.6f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
        
        # Check spectral radius at specific epochs
        if (epoch + 1) in epochs_to_check:
            sr = save_weight_checkpoint(policy, epoch + 1, save_dir)
            spectral_radii.append((epoch + 1, sr))
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            model_name = 'best_rnn_robust.pth'
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
                    'rho_target': rho_target if clamp_spectral_radius else None,
                    # NEW: Robustness config
                    'robustness_training': True,
                    'noise_std': noise_std,
                    'noise_prob': noise_prob,
                    'noise_type': noise_type,
                    'progressive_noise': progressive_noise
                }
            }, os.path.join(save_dir, model_name))
            print(f"   💾 New best robust model saved! Loss: {avg_loss:.6f}")
        
        # Save checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0:
            checkpoint_name = f'rnn_robust_checkpoint_epoch_{epoch+1}.pth'
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
                    'rho_target': rho_target if clamp_spectral_radius else None,
                    'robustness_training': True,
                    'noise_std': noise_std,
                    'noise_prob': noise_prob,
                    'noise_type': noise_type,
                    'progressive_noise': progressive_noise
                }
            }, os.path.join(save_dir, checkpoint_name))
    
    # Enhanced plotting with robustness metrics
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Loss curve
    plt.subplot(2, 3, 1)
    plt.plot(losses)
    plt.title('Robust RNN Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.grid(True)
    
    # Plot 2: Gradient norms
    plt.subplot(2, 3, 2)
    plt.plot(all_grad_norms)
    plt.title('Gradient Norms During Training')
    plt.xlabel('Update Step')
    plt.ylabel('Gradient Norm')
    plt.yscale('log')
    plt.grid(True)
    
    # Plot 3: Spectral radius evolution
    plt.subplot(2, 3, 3)
    epochs, radii = zip(*spectral_radii)
    plt.plot(epochs, radii, 'o-', linewidth=2)
    plt.title('Spectral Radius Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Spectral Radius')
    if clamp_spectral_radius:
        plt.axhline(y=rho_target, color='red', linestyle='--', alpha=0.7, label=f'Target ρ = {rho_target}')
        plt.legend()
    plt.grid(True)
    
    # Plot 4: Noise schedule (if progressive)
    plt.subplot(2, 3, 4)
    if progressive_noise:
        noise_stds, noise_probs = zip(*noise_schedule)
        plt.plot(noise_stds, label='Noise std', alpha=0.8)
        plt.plot(noise_probs, label='Noise prob', alpha=0.8)
        plt.title('Progressive Noise Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, f'Fixed noise:\nstd={noise_std:.3f}\nprob={noise_prob:.2f}', 
                transform=plt.gca().transAxes, ha='center', va='center', fontsize=12)
        plt.title('Noise Configuration')
    
    # Plot 5: Gradient norm distribution
    plt.subplot(2, 3, 5)
    plt.hist(all_grad_norms, bins=50, alpha=0.7)
    plt.title('Gradient Norm Distribution')
    plt.xlabel('Gradient Norm')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True)
    
    # Plot 6: Loss smoothed
    plt.subplot(2, 3, 6)
    # Moving average for smoother visualization
    window = min(20, len(losses) // 10)
    if window > 1:
        smoothed_losses = np.convolve(losses, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(losses)), smoothed_losses, label=f'Smoothed (window={window})')
        plt.plot(losses, alpha=0.3, label='Raw')
        plt.legend()
    else:
        plt.plot(losses)
    plt.title('Training Loss (Smoothed)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'robust_training_metrics.png'), dpi=150)
    plt.close()
    
    # Print comprehensive summary
    print("\n" + "="*60)
    print("🛡️  ROBUST TRAINING COMPLETED!")
    print("="*60)
    print(f"Best loss: {best_loss:.6f}")
    print(f"Models saved in: {save_dir}")
    
    print(f"\n📊 ROBUSTNESS SUMMARY:")
    print(f"   Noise type: {noise_type}")
    print(f"   Target noise std: {noise_std:.4f}")
    print(f"   Target noise prob: {noise_prob:.2f}")
    print(f"   Progressive schedule: {progressive_noise}")
    
    if clamp_spectral_radius:
        print(f"\n🔧 SPECTRAL RADIUS SUMMARY:")
        final_sr = spectral_radii[-1][1] if spectral_radii else "unknown"
        print(f"   Target ρ: {rho_target}")
        print(f"   Final ρ: {final_sr:.4f}")
    
    print(f"\n💡 EVALUATION NOTES:")
    print(f"   - Use 'selective_vec_normalize.pkl' for evaluation")
    print(f"   - Model trained with noise injection for robustness")
    print(f"   - Should handle distributional shift better than baseline")
    
    return policy

if __name__ == "__main__":
    # Train robust RNN with noise injection
    train_robust_rnn(
        demonstrations_path="sac_demonstrations_50steps_successful_selective5.pkl",
        save_dir="./rnn_robust_noise_injection",
        selective_norm_path="selective_vec_normalize3.pkl",
        num_epochs=400,
        batch_size=16,
        sequence_length=50,
        learning_rate=1e-3,
        recurrent_lr_factor=0.1,
        hidden_size=64,
        tau_mem=1.0,
        clamp_spectral_radius=True,
        rho_target=0.95,
        # Robustness parameters
        noise_std=0.015,        # Small noise to avoid corrupting data too much
        noise_prob=0.4,         # Apply to 40% of batches
        noise_type='gaussian',  # Standard Gaussian noise
        progressive_noise=True  # Start small, increase gradually
    )