import torch
import matplotlib.pyplot as plt
import os
import numpy as np

def extract_losses_from_checkpoints(checkpoint_dirs):
    """Extract loss values from multiple training checkpoint directories"""
    
    all_losses = {}
    
    for i, checkpoint_dir in enumerate(checkpoint_dirs):
        losses = []
        epochs = []
        
        # Look for checkpoint files
        checkpoint_files = []
        if os.path.exists(checkpoint_dir):
            for file in os.listdir(checkpoint_dir):
                if file.startswith('rnn_checkpoint_epoch_') and file.endswith('.pth'):
                    checkpoint_files.append(file)
        
        # Sort by epoch number
        checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        # Load each checkpoint and extract loss
        for checkpoint_file in checkpoint_files:
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                epoch = checkpoint['epoch']
                loss = checkpoint['loss']
                
                epochs.append(epoch + 1)  # Convert to 1-indexed
                losses.append(loss)
                
            except Exception as e:
                print(f"Error loading {checkpoint_file}: {e}")
                continue
        
        # Determine label from directory name
        if '25hidden' in checkpoint_dir:
            label = '25 Hidden Units'
        elif '64hidden' in checkpoint_dir:
            label = '64 Hidden Units'
        elif '128hidden' in checkpoint_dir:
            label = '128 Hidden Units'
        else:
            label = f'Training {i+1}'
        
        # Manual early epoch data - ADD YOUR VALUES HERE
        if label == '25 Hidden Units':
            manual_epochs = [0, 5, 10, 15]
            manual_losses = [0.14, 0.045, 0.029, 0.024]  # REPLACE WITH YOUR VALUES
            epochs = manual_epochs + epochs
            losses = manual_losses + losses
            
        elif label == '64 Hidden Units':
            manual_epochs = [0, 5, 10, 15]
            manual_losses = [0.14, 0.04, 0.022, 0.018]  # REPLACE WITH YOUR VALUES
            epochs = manual_epochs + epochs
            losses = manual_losses + losses
            
        elif label == '128 Hidden Units':
            manual_epochs = [0, 5, 10, 15]
            manual_losses = [0.13, 0.035, 0.02, 0.015]  # REPLACE WITH YOUR VALUES
            epochs = manual_epochs + epochs
            losses = manual_losses + losses
        
        # Also try to load the best model
        best_model_path = os.path.join(checkpoint_dir, 'best_rnn_full.pth')
        if os.path.exists(best_model_path):
            try:
                best_checkpoint = torch.load(best_model_path, map_location='cpu')
                print(f"{label} - Best loss: {best_checkpoint['loss']:.6f} at epoch {best_checkpoint['epoch']+1}")
            except:
                pass
        
        if losses:
            all_losses[label] = {'epochs': epochs, 'losses': losses}
            print(f"{label}: Found {len(losses)} checkpoints (epochs {min(epochs)}-{max(epochs)})")
        else:
            print(f"{label}: No valid checkpoints found in {checkpoint_dir}")
    
    return all_losses

def plot_combined_losses(all_losses, save_path='combined_training_losses.png'):
    """Plot MSE losses from multiple training runs"""
    
    plt.figure(figsize=(14, 10))  # Made figure bigger
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    linestyles = ['-', '-', '-', '--', '-.', ':']
    
    for i, (training_name, data) in enumerate(all_losses.items()):
        epochs = data['epochs']
        losses = data['losses']
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        
        # Plot the complete line
        plt.plot(epochs, losses, color=color, linewidth=3, 
                markersize=6, label=training_name, alpha=0.8, 
                linestyle=linestyle, marker='o')
        
        # Add final loss annotation with bigger font
        if losses:
            final_loss = losses[-1]
            final_epoch = epochs[-1]
            plt.annotate(f'{final_loss:.4f}', 
                        xy=(final_epoch, final_loss), 
                        xytext=(12, 12), textcoords='offset points',
                        fontsize=16, color=color, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
    
    plt.xlabel('Epoch', fontsize=24, fontweight='bold')
    plt.ylabel('MSE Loss', fontsize=24, fontweight='bold')
    plt.title('RNN Training Loss Comparison: Different Hidden Layer Sizes', 
              fontsize=26, fontweight='bold', pad=20)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=18, loc='upper right')
    
    # Make tick labels bigger
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=16)
    
    # Add some styling
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Combined loss plot saved to: {save_path}")

# Use only the main checkpoint directories
checkpoint_directories = [
    "./rnn_test_25hidden_spectral_clamped2",
    "./rnn_test_64hidden_spectral_clamped2", 
    "./rnn_test_128hidden_spectral_clamped2"
]

# Extract losses from all training runs
all_losses = extract_losses_from_checkpoints(checkpoint_directories)

# Plot combined losses
if all_losses:
    plot_combined_losses(all_losses)
    
    # Print summary statistics
    print("\n📊 Training Summary:")
    print("=" * 60)
    for training_name, data in all_losses.items():
        losses = data['losses']
        epochs = data['epochs']
        if losses:
            min_loss = min(losses)
            final_loss = losses[-1]
            min_loss_epoch = epochs[losses.index(min_loss)]
            initial_loss = losses[0]
            print(f"{training_name}:")
            print(f"  Initial loss: {initial_loss:.6f} (epoch {epochs[0]})")
            print(f"  Final loss:   {final_loss:.6f} (epoch {epochs[-1]})")
            print(f"  Best loss:    {min_loss:.6f} (epoch {min_loss_epoch})")
            if len(losses) > 1:
                print(f"  Improvement: {((initial_loss - final_loss) / initial_loss * 100):.1f}%")
                print(f"  Loss reduction: {initial_loss/final_loss:.1f}x")
            print()
else:
    print("❌ No valid checkpoint data found!")