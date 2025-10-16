import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

def visualize_long_term_predictions(initial_steps, predictions, ground_truth, 
                                   sample_idx=0, max_frames=50, fps=8, 
                                   save_dir='./long_term_results/'):
    """
    Create comprehensive visualization of long-term predictions with 4-panel layout
    
    Args:
        initial_steps: [batch, t_in, n_vars, H, W] - your input (first t_in timesteps)
        predictions: [batch, T_pred, n_vars, H, W] - model predictions
        ground_truth: [batch, T_truth, n_vars, H, W] - true evolution 
        sample_idx: which sample from batch to visualize
        max_frames: how many timesteps to show in animation
        fps: frames per second for GIF
        save_dir: directory to save the GIF
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to CPU and numpy if needed
    if hasattr(initial_steps, 'cpu'):
        initial_steps = initial_steps.cpu()
    if hasattr(predictions, 'cpu'):
        predictions = predictions.cpu() 
    if hasattr(ground_truth, 'cpu'):
        ground_truth = ground_truth.cpu()
    
    # Extract data for one sample
    initial = initial_steps[0].numpy()  # [t_in, n_vars, H, W]
    pred = predictions[0].numpy()       # [T_pred, n_vars, H, W]
    truth = ground_truth[0].numpy()     # [T_truth, n_vars, H, W]
    
    print(f"Initial shape: {initial.shape}")
    print(f"Predictions shape: {pred.shape}")
    print(f"Ground truth shape: {truth.shape}")
    
    # Use minimum available timesteps
    available_frames = min(pred.shape[0], truth.shape[0], max_frames)
    print(f"Animating {available_frames} frames")
    
    # Truncate to matching length
    pred = pred[:available_frames]
    truth = truth[:available_frames]
    
    # Use first variable (channel 0)
    var_idx = 0
    
    initial_var = initial[:, var_idx]  # [t_in, H, W]
    pred_var = pred[:, var_idx]        # [T, H, W] 
    truth_var = truth[:, var_idx]      # [T, H, W]
    
    # Calculate error
    error = np.abs(pred_var - truth_var)
    
    # Global color scale for consistent visualization
    all_data = np.concatenate([
        initial_var.flatten(),
        pred_var.flatten(), 
        truth_var.flatten()
    ])
    vmin, vmax = np.percentile(all_data, [2, 98])
    
    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Long-term Prediction Analysis - Sample {sample_idx}', 
                 fontsize=16, fontweight='bold')
    
    # Define subplot properties
    subplot_config = [
        (axes[0, 0], 'Input Context (Last Frame)', 'RdBu_r', True),   # Top-left: input
        (axes[0, 1], 'Model Prediction', 'RdBu_r', False),            # Top-right: prediction  
        (axes[1, 0], 'Ground Truth', 'RdBu_r', False),                # Bottom-left: truth
        (axes[1, 1], 'Absolute Error', 'Reds', False)                 # Bottom-right: error
    ]
    
    ims = []
    
    # Initialize each subplot
    for i, (ax, title, cmap, is_static) in enumerate(subplot_config):
        if is_static:  # Input context - show last input frame (static)
            data = initial_var[-1]  # Last input timestep
            im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, 
                          origin='lower', aspect='equal')
        elif title == 'Absolute Error':  # Error plot
            im = ax.imshow(error[0], vmin=0, vmax=np.max(error), cmap=cmap,
                          origin='lower', aspect='equal')
        else:  # Prediction or ground truth
            if title == 'Model Prediction':
                data = pred_var[0] 
            else:  # Ground Truth
                data = truth_var[0]
            im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap,
                          origin='lower', aspect='equal')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X Grid Points')
        ax.set_ylabel('Y Grid Points')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        if title == 'Absolute Error':
            cbar.set_label('|Prediction - Truth|', fontsize=10)
        else:
            cbar.set_label('Field Value', fontsize=10)
        
        ims.append(im)
    
    # Time indicator at bottom of figure
    time_text = fig.text(0.5, 0.02, '', ha='center', fontsize=14, fontweight='bold')
    
    def animate(frame):
        # Update only the dynamic panels:
        # ims[0] = input context (static, no update needed)
        # ims[1] = prediction (update)
        # ims[2] = ground truth (update) 
        # ims[3] = error (update)
        
        ims[1].set_array(pred_var[frame])      # Update prediction
        ims[2].set_array(truth_var[frame])     # Update ground truth
        ims[3].set_array(error[frame])         # Update error
        
        # Update error colorbar limits for current frame
        ims[3].set_clim(0, np.max(error[frame]))
        
        # Update time indicator
        time_text.set_text(f'Prediction Timestep: {frame+1}/{available_frames} | '
                          f'Input Context: {initial.shape[0]} timesteps')
        
        return ims + [time_text]
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=available_frames,
                                  interval=1000//fps, blit=True, repeat=True)
    
    # Save GIF
    gif_path = f"{save_dir}/long_term_prediction_sample_{sample_idx}.gif"
    print(f"Saving long-term prediction GIF: {gif_path}")
    
    try:
        anim.save(gif_path, writer='pillow', fps=fps, dpi=100)
        print(f"Successfully saved: {gif_path}")
    except Exception as e:
        print(f"Error saving GIF: {e}")
    
    plt.tight_layout()
    plt.show()
    
    return anim

# ============= ESSENTIAL TRAINING VISUALIZATION FUNCTIONS =============

def visualize_predictions(model, val_loader, grid, data_mean, data_std, epoch, device, 
                         n_samples=3, save_dir='training_visualizations'):
    """
    Static visualization showing last timestep comparison: input/true/prediction
    Layout: n_samples x 3 grid (input/true/pred for last timestep)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        # Get one batch from validation set
        batch = next(iter(val_loader))
        x_batch, y_batch = batch["input_fields"], batch["output_fields"]
        x_batch = x_batch.permute(0, 1, 4, 2, 3)
        y_batch = y_batch.permute(0, 1, 4, 2, 3)   
        
        # Move to device FIRST, then normalize
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        x_batch_norm = (x_batch - data_mean) / data_std
        y_batch_norm = (y_batch - data_mean) / data_std
        
        # Create batch grid
        batch_size = x_batch_norm.shape[0]
        batch_grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # Get predictions
        pred_batch = model.predict(x_batch_norm, n_steps=y_batch_norm.shape[1], grid=batch_grid)
        
        # Keep normalized for fair comparison
        x_batch_viz = x_batch_norm.cpu().numpy()
        y_batch_viz = y_batch_norm.cpu().numpy()
        pred_batch_viz = pred_batch.cpu().numpy()
        
        # Select random samples
        n_samples = min(n_samples, batch_size)
        sample_indices = np.random.choice(batch_size, n_samples, replace=False)
        
        # Create figure
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        fig.suptitle(f'Epoch {epoch} - Last Timestep Comparison - Variable 0 (Density)', fontsize=16)
        
        # Variable 0 (density)
        var = 0
        
        for row_idx, sample_idx in enumerate(sample_indices):
            # Last timesteps
            last_input_t = x_batch_viz.shape[1] - 1
            last_pred_t = y_batch_viz.shape[1] - 1
            
            input_data = x_batch_viz[sample_idx, last_input_t, var]
            true_data = y_batch_viz[sample_idx, last_pred_t, var]
            pred_data = pred_batch_viz[sample_idx, last_pred_t, var]
            
            # Global color scale
            all_data = [input_data, true_data, pred_data]
            vmin = min(d.min() for d in all_data)
            vmax = max(d.max() for d in all_data)
            
            # Plot three comparisons
            images_data = [
                (input_data, f'Input t={last_input_t}'),
                (true_data, f'True t={last_pred_t}'),
                (pred_data, f'Pred t={last_pred_t}')
            ]
            
            for col_idx, (data, title) in enumerate(images_data):
                ax = axes[row_idx, col_idx]
                im = ax.imshow(data, cmap='plasma', aspect='equal', origin='lower',
                             vmin=vmin, vmax=vmax)
                ax.set_title(f'Sample {sample_idx}: {title}')
                ax.set_xticks([])
                ax.set_yticks([])
                plt.colorbar(im, ax=ax, shrink=0.8)
            
            # Calculate MSE for this sample
            mse = np.mean((true_data - pred_data)**2)
            print(f"  Sample {sample_idx} MSE: {mse:.6f}")
        
        plt.tight_layout()
        
        # Save figure
        save_path = f"{save_dir}/epoch_{epoch}_predictions.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()




def create_gif_visualization(model, val_loader, grid, data_mean, data_std, epoch, device, n_samples=4):
    """
    Create animated GIF showing input->prediction->groundtruth evolution
    Only use occasionally due to computational cost
    """
    os.makedirs('training_visualizations', exist_ok=True)
    
    model.eval()
    
    # Get random samples
    all_samples = []
    with torch.no_grad():
        for batch in val_loader:
            x_batch, y_batch = batch["input_fields"], batch["output_fields"]
            x_batch = x_batch.permute(0, 1, 4, 2, 3)
            y_batch = y_batch.permute(0, 1, 4, 2, 3)            
            
            # Move to device FIRST, then normalize
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            x_batch = (x_batch - data_mean) / data_std
            y_batch = (y_batch - data_mean) / data_std
            
            for i in range(x_batch.shape[0]):
                all_samples.append((x_batch[i:i+1], y_batch[i:i+1]))
            
            if len(all_samples) >= 20:
                break
    
    # Select samples
    selected_samples = random.sample(all_samples, min(n_samples, len(all_samples)))
    
    # Generate predictions
    predictions = []
    inputs = []
    ground_truths = []
    
    with torch.no_grad():
        for x_sample, y_sample in selected_samples:
            x_sample = x_sample.to(device)
            y_sample = y_sample.to(device)
            
            batch_grid = grid.unsqueeze(0).expand(1, -1, -1, -1)
            pred = model.predict(x_sample, n_steps=y_sample.shape[1], grid=batch_grid)
            
            # Denormalize for visualization
            x_denorm = (x_sample * data_std + data_mean).cpu().numpy()
            pred_denorm = (pred * data_std + data_mean).cpu().numpy()
            y_denorm = (y_sample * data_std + data_mean).cpu().numpy()
            
            inputs.append(x_denorm[0])
            predictions.append(pred_denorm[0])
            ground_truths.append(y_denorm[0])
    
    # Create GIF
    n_timesteps = predictions[0].shape[0]
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(f'Epoch {epoch} - Evolution', fontsize=16)
    
    # Set titles
    for i in range(n_samples):
        if i == 0:
            axes[i, 0].set_title('Input (Last)', fontsize=12)
            axes[i, 1].set_title('Prediction', fontsize=12)
            axes[i, 2].set_title('Ground Truth', fontsize=12)
        axes[i, 0].set_ylabel(f'Sample {i+1}', fontsize=12)
    
    # Initialize images
    images = []
    for i in range(n_samples):
        row_images = []
        for j in range(3):
            if j == 0:
                data = inputs[i][-1, 0]  # Last input frame
            else:
                data = np.zeros_like(predictions[i][0, 0])
            
            im = axes[i, j].imshow(data, cmap='RdBu_r', animated=True)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            row_images.append(im)
        images.append(row_images)
    
    def animate(frame):
        for i in range(n_samples):
            # Input stays static
            images[i][0].set_array(inputs[i][-1, 0])
            # Prediction and ground truth evolve
            images[i][1].set_array(predictions[i][frame, 0])
            images[i][2].set_array(ground_truths[i][frame, 0])
            
            # Update color limits
            for j in range(3):
                if j == 0:
                    data = inputs[i][-1, 0]
                elif j == 1:
                    data = predictions[i][frame, 0]
                else:
                    data = ground_truths[i][frame, 0]
                
                vmin, vmax = np.percentile(data, [5, 95])
                if vmin == vmax:
                    vmin -= 0.1
                    vmax += 0.1
                images[i][j].set_clim(vmin, vmax)
        
        return [img for row in images for img in row]
    
    # Create and save animation
    anim = animation.FuncAnimation(
        fig, animate, frames=n_timesteps, 
        interval=200, blit=True, repeat=True
    )
    
    gif_path = f'training_visualizations/epoch_{epoch}_evolution.gif'
    try:
        anim.save(gif_path, writer='pillow', fps=5, dpi=80)  # Lower DPI for speed
        print(f"  GIF saved: {gif_path}")
    except Exception as e:
        print(f"  GIF save failed: {e}")
    finally:
        plt.close(fig)


def plot_gif(model, val_loader, grid, data_mean, data_std, N, device, 
             save_path='training_visualizations', epoch=None, prefix='sample'):
    """
    Create N simple GIFs - lightweight version for frequent use
    """
    os.makedirs(save_path, exist_ok=True)
    
    model.eval()
    
    # Collect samples
    all_samples = []
    with torch.no_grad():
        for batch in val_loader:
            x_batch, y_batch = batch["input_fields"], batch["output_fields"]
            x_batch = x_batch.permute(0, 1, 4, 2, 3)
            y_batch = y_batch.permute(0, 1, 4, 2, 3)   
            
            # Move to device FIRST, then normalize  
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            x_batch = (x_batch - data_mean) / data_std
            y_batch = (y_batch - data_mean) / data_std
            
            if len(all_samples) >= 50:
                break
    
    # Select N samples
    N_available = min(N, len(all_samples))
    selected_samples = random.sample(all_samples, N_available)
    
    for sample_idx, (x_sample, y_sample) in enumerate(selected_samples):
        x_sample = x_sample.to(device)
        y_sample = y_sample.to(device)
        
        with torch.no_grad():
            batch_grid = grid.unsqueeze(0).expand(1, -1, -1, -1)
            pred_sample = model.predict(x_sample, n_steps=y_sample.shape[1], grid=batch_grid)
        
        # Denormalize
        x_denorm = (x_sample * data_std + data_mean).cpu().numpy()
        pred_denorm = (pred_sample * data_std + data_mean).cpu().numpy()
        y_denorm = (y_sample * data_std + data_mean).cpu().numpy()
        
        # Simple 1x3 layout
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Sample {sample_idx + 1}', fontsize=14)
        
        axes[0].set_title('Input')
        axes[1].set_title('Prediction') 
        axes[2].set_title('Ground Truth')
        
        # Initialize
        n_timesteps = pred_denorm.shape[1]
        images = []
        
        for j in range(3):
            if j == 0:
                data = x_denorm[0, -1, 0]  # Last input
            else:
                data = np.zeros_like(y_denorm[0, 0, 0])
            
            im = axes[j].imshow(data, cmap='plasma', animated=True)
            axes[j].set_xticks([])
            axes[j].set_yticks([])
            images.append(im)
        
        def animate_simple(frame):
            images[0].set_array(x_denorm[0, -1, 0])  # Static input
            images[1].set_array(pred_denorm[0, frame, 0])  # Prediction
            images[2].set_array(y_denorm[0, frame, 0])  # Ground truth
            return images
        
        anim = animation.FuncAnimation(
            fig, animate_simple, frames=n_timesteps,
            interval=200, blit=True, repeat=True
        )
        
        # Save
        epoch_str = f"epoch_{epoch}_" if epoch is not None else ""
        filename = f"{prefix}_{epoch_str}{sample_idx + 1:03d}.gif"
        gif_path = os.path.join(save_path, filename)
        
        try:
            anim.save(gif_path, writer='pillow', fps=5, dpi=60)  # Very low DPI for speed
        except Exception as e:
            print(f"  GIF save failed: {e}")
        finally:
            plt.close(fig)
    
    print(f"Created {N_available} simple GIFs")


def create_error_evolution_plot(predictions, ground_truth, save_dir='./long_term_results/'):
    """
    Create plots showing how prediction error evolves over time
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate errors over time for all samples
    pred_np = predictions.cpu().numpy()  # [batch, T_pred, n_vars, H, W]
    truth_np = ground_truth.cpu().numpy()  # [batch, T_truth, n_vars, H, W]
    

    
    # Use the minimum number of timesteps available
    n_timesteps = min(pred_np.shape[1], truth_np.shape[1])
    
    # Truncate to matching length
    pred_np = pred_np[:, :n_timesteps]
    truth_np = truth_np[:, :n_timesteps]
    
    timesteps = np.arange(1, n_timesteps + 1)
    
    mae_evolution = []
    rmse_evolution = []
    max_error_evolution = []
    
    for t in range(n_timesteps):
        # Calculate errors at timestep t across all samples
        error_t = np.abs(pred_np[:, t] - truth_np[:, t])
        
        mae_evolution.append(np.mean(error_t))
        rmse_evolution.append(np.sqrt(np.mean((pred_np[:, t] - truth_np[:, t])**2)))
        max_error_evolution.append(np.max(error_t))
    
    # Create error evolution plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Long-term Prediction Error Analysis', fontsize=16, fontweight='bold')
    
    # MAE evolution
    axes[0, 0].plot(timesteps, mae_evolution, 'r-', linewidth=2, label='MAE')
    axes[0, 0].set_xlabel('Prediction Timestep')
    axes[0, 0].set_ylabel('Mean Absolute Error')
    axes[0, 0].set_title('MAE Evolution')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # RMSE evolution
    axes[0, 1].plot(timesteps, rmse_evolution, 'b-', linewidth=2, label='RMSE')
    axes[0, 1].set_xlabel('Prediction Timestep')
    axes[0, 1].set_ylabel('Root Mean Square Error')
    axes[0, 1].set_title('RMSE Evolution')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Max error evolution
    axes[1, 0].plot(timesteps, max_error_evolution, 'g-', linewidth=2, label='Max Error')
    axes[1, 0].set_xlabel('Prediction Timestep')
    axes[1, 0].set_ylabel('Maximum Error')
    axes[1, 0].set_title('Maximum Error Evolution')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Error distribution at different timesteps
    selected_times = [0, n_timesteps//4, n_timesteps//2, 3*n_timesteps//4, n_timesteps-1]
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    for i, (t, color) in enumerate(zip(selected_times, colors)):
        error_dist = np.abs(pred_np[:, t] - truth_np[:, t]).flatten()
        axes[1, 1].hist(error_dist, bins=50, alpha=0.6, color=color, 
                       label=f't={t+1}', density=True)
    
    axes[1, 1].set_xlabel('Absolute Error')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Error Distribution at Different Times')
    axes[1, 1].legend()
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/error_evolution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_gif_comparison(model, val_loader, grid, data_mean, data_std, N, device,
                       save_path='training_visualizations', epoch=None):
    """
    Alternative version that creates comparison GIFs with metrics overlay
    
    Args:
        model: Trained model  
        val_loader: Validation data loader
        grid: Spatial grid for the model
        data_mean: Data normalization mean
        data_std: Data normalization std  
        N: Number of random samples
        device: torch device
        save_path: Directory to save GIFs
        epoch: Current epoch
    """
    # Create directory
    os.makedirs(save_path, exist_ok=True)
    
    model.eval()
    
    # Collect samples
    print(f"Collecting samples for comparison GIFs...")
    all_samples = []
    with torch.no_grad():
        for batch in val_loader:
            x_batch, y_batch = batch["input_fields"], batch["output_fields"]
            x_batch = x_batch.permute(0, 1, 4, 2, 3)
            y_batch = y_batch.permute(0, 1, 4, 2, 3)            

            
            for i in range(x_batch.shape[0]):
                all_samples.append((x_batch[i:i+1], y_batch[i:i+1]))
            
            if len(all_samples) >= 50:
                break
    
    # Select N samples
    N_available = min(N, len(all_samples))
    selected_samples = random.sample(all_samples, N_available)
    
    for sample_idx, (x_sample, y_sample) in enumerate(selected_samples):
        print(f"Creating comparison GIF {sample_idx + 1}/{N_available}...")
        
        # Generate prediction
        x_sample = x_sample.to(device)
        y_sample = y_sample.to(device)
        
        with torch.no_grad():
            batch_grid = grid.unsqueeze(0).expand(1, -1, -1, -1)
            pred_sample = model.predict(x_sample, n_steps=y_sample.shape[1], grid=batch_grid)
        
        # Denormalize
        x_denorm = (x_sample * data_std + data_mean).cpu().numpy()[0]
        pred_denorm = (pred_sample * data_std + data_mean).cpu().numpy()[0]
        y_denorm = (y_sample * data_std + data_mean).cpu().numpy()[0]
        
        # Calculate metrics for each timestep
        timestep_errors = []
        for t in range(pred_denorm.shape[0]):
            mse = np.mean((pred_denorm[t] - y_denorm[t])**2)
            mae = np.mean(np.abs(pred_denorm[t] - y_denorm[t]))
            timestep_errors.append({'mse': mse, 'mae': mae})
        
        # Create figure with additional space for metrics
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.3], hspace=0.3, wspace=0.2)
        
        # Create subplots for channels
        axes = []
        for row in range(2):
            row_axes = []
            for col in range(3):
                ax = fig.add_subplot(gs[row, col])
                row_axes.append(ax)
            axes.append(row_axes)
        
        # Metrics subplot
        metrics_ax = fig.add_subplot(gs[2, :])
        
        # Set titles and labels
        fig.suptitle(f'Sample {sample_idx + 1} - Detailed Comparison', fontsize=16)
        axes[0][0].set_title('Input (Last Frame)', fontsize=12)
        axes[0][1].set_title('Ground Truth', fontsize=12)
        axes[0][2].set_title('Prediction', fontsize=12)
        
        axes[0][0].set_ylabel('Channel 1', fontsize=12)
        axes[1][0].set_ylabel('Channel 2', fontsize=12)
        
        # Initialize images and metrics plot
        images = []
        n_channels = min(2, pred_denorm.shape[1])
        n_timesteps = pred_denorm.shape[0]
        
        for channel in range(2):
            row_images = []
            cmap = ['RdBu_r', 'plasma'][channel]
            
            for col in range(3):
                if channel < n_channels:
                    if col == 0:
                        init_data = x_denorm[-1, channel]
                    else:
                        init_data = np.zeros_like(y_denorm[0, channel])
                else:
                    init_data = np.zeros_like(y_denorm[0, 0])
                
                im = axes[channel][col].imshow(init_data, cmap=cmap, animated=True)
                axes[channel][col].set_xticks([])
                axes[channel][col].set_yticks([])
                row_images.append(im)
            
            images.append(row_images)
        
        # Initialize metrics plot
        times = range(n_timesteps)
        mse_line, = metrics_ax.plot(times, [e['mse'] for e in timestep_errors], 'b-', label='MSE')
        mae_line, = metrics_ax.plot(times, [e['mae'] for e in timestep_errors], 'r-', label='MAE')
        current_time_line = metrics_ax.axvline(x=0, color='green', linestyle='--', label='Current Time')
        
        metrics_ax.set_xlabel('Time Step')
        metrics_ax.set_ylabel('Error')
        metrics_ax.set_title('Prediction Error Over Time')
        metrics_ax.legend()
        metrics_ax.grid(True, alpha=0.3)
        
        def animate_comparison(frame):
            updated = []
            
            # Update images
            for channel in range(2):
                for col in range(3):
                    if channel < n_channels:
                        if col == 0:
                            data = x_denorm[-1, channel]
                        elif col == 1:
                            data = y_denorm[frame, channel]
                        else:
                            data = pred_denorm[frame, channel]
                    else:
                        data = np.zeros_like(y_denorm[0, 0])
                    
                    images[channel][col].set_array(data)
                    if np.any(data != 0):
                        vmin, vmax = np.percentile(data, [2, 98])
                        if vmin == vmax:
                            vmin -= 0.1
                            vmax += 0.1
                        images[channel][col].set_clim(vmin, vmax)
                    
                    updated.append(images[channel][col])
            
            # Update metrics line
            current_time_line.set_xdata([frame])
            updated.extend([current_time_line])
            
            return updated
        
        # Create and save animation
        anim = animation.FuncAnimation(
            fig, animate_comparison, frames=n_timesteps,
            interval=300, blit=True, repeat=True
        )
        
        epoch_str = f"epoch_{epoch}_" if epoch is not None else ""
        filename = f"comparison_{epoch_str}sample_{sample_idx + 1:03d}.gif"
        gif_path = os.path.join(save_path, filename)
        
        try:
            anim.save(gif_path, writer='pillow', fps=3, dpi=80)
            print(f"  ✓ Comparison GIF saved: {gif_path}")
        except Exception as e:
            print(f"  ✗ Failed to save comparison GIF: {e}")
        finally:
            plt.close(fig)
    
    print(f"Completed creating {N_available} comparison GIFs")


def create_long_term_predictions(model, full_data, grid, data_mean, data_std, device,
                                 output_dir, n_samples=20, t_in=10):
    """Create long-term predictions for multiple samples and visualize them."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Creating long-term predictions for {n_samples} samples...")
    model.eval()

    # Get data
    data = full_data['data']
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    
    n_trajectories, n_timesteps = data.shape[0], data.shape[1]
    max_pred_steps = n_timesteps - t_in
    
    # Select samples
    sample_indices = np.random.choice(n_trajectories, min(n_samples, n_trajectories), replace=False)

    # Ensure tensors on device
    data_mean = torch.as_tensor(data_mean, dtype=torch.float32).to(device)
    data_std = torch.as_tensor(data_std, dtype=torch.float32).to(device)
    grid = grid.to(device)

    with torch.no_grad():
        for idx, sample_idx in enumerate(sample_indices):
            print(f"\nProcessing sample {idx+1}/{len(sample_indices)} (trajectory {sample_idx})...")
            
            trajectory = data[sample_idx]
            if isinstance(trajectory, np.ndarray):
                trajectory = torch.from_numpy(trajectory)
            
            # Check if channels-last format and fix
            #if trajectory.shape[-1] <= model.n_var and trajectory.shape[-2] > model.n_var:
            trajectory = trajectory.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]

            # Split trajectory
            initial_steps = trajectory[:t_in]
            ground_truth = trajectory[t_in:]

            # Prepare input and normalize
            x_input = initial_steps.unsqueeze(0).float().to(device)
            x_input_norm = (x_input - data_mean) / data_std

            # Create batch grid
            batch_grid = grid.unsqueeze(0).expand(1, -1, -1, -1)

            # Generate predictions
            print(f"  Generating {max_pred_steps} timestep predictions...")
            predictions = model.predict(x_input_norm, n_steps=max_pred_steps, grid=batch_grid)
            predictions_denorm = predictions * data_std + data_mean

            # Convert to numpy
            pred_np = predictions_denorm.cpu().numpy()
            initial_np = initial_steps.unsqueeze(0).cpu().numpy()
            truth_np = ground_truth.unsqueeze(0).cpu().numpy()

            # Visualize
            print(f"  Creating visualization...")
            try:
                visualize_long_term_predictions(
                    initial_steps=torch.from_numpy(initial_np),
                    predictions=torch.from_numpy(pred_np),
                    ground_truth=torch.from_numpy(truth_np),
                    sample_idx=idx+1,
                    max_frames=min(50, max_pred_steps),
                    fps=8,
                    save_dir=output_dir
                )
                print(f" Visualization saved")
            except Exception as e:
                print(f" Visualization error: {e}")

            # Calculate metrics
            pred_single = pred_np[0]
            truth_single = truth_np[0]
            n_compare = min(pred_single.shape[0], truth_single.shape[0])

            mae_list, rmse_list = [], []
            for t in range(n_compare):
                diff = pred_single[t] - truth_single[t]
                mae_list.append(np.mean(np.abs(diff)))
                rmse_list.append(np.sqrt(np.mean(diff**2)))

            # Save metrics
            metrics_path = os.path.join(output_dir, f'sample_{sample_idx}_metrics.txt')
            with open(metrics_path, 'w') as f:
                f.write(f"Long-term Prediction Metrics - Sample {sample_idx}\n{'='*60}\n\n")
                f.write(f"Input timesteps: {t_in}\nPredicted timesteps: {n_compare}\n\n")
                f.write(f"Average MAE: {np.mean(mae_list):.6f}\n")
                f.write(f"Average RMSE: {np.mean(rmse_list):.6f}\n")
                f.write(f"Final MAE: {mae_list[-1]:.6f}\nFinal RMSE: {rmse_list[-1]:.6f}\n\n")
                f.write("Per-timestep errors:\n" + "-"*60 + "\n")
                for t in range(n_compare):
                    f.write(f"t={t+1:3d}: MAE={mae_list[t]:.6f}, RMSE={rmse_list[t]:.6f}\n")

            print(f" Metrics saved")

    print(f"\n{'='*60}\nLong-term prediction complete!\nResults: {output_dir}\n{'='*60}\n")