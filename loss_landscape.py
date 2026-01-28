"""Visualization of loss landscapes through multiple network training runs.

This module trains multiple neural networks with different random initializations
on the same dataset and visualizes their weight trajectories in a reduced dimensional
space using PCA.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.datasets import make_moons
import torch

from ml_utils import SimpleNN, train_network, create_dataloader


def generate_dataset(n_samples=200, noise=0.2, random_state=42):
    """Generate a synthetic moon-shaped dataset.
    
    Args:
        n_samples: Number of samples to generate (default: 200)
        noise: Standard deviation of Gaussian noise (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        
    Returns:
        tuple: (X, y) where X is features and y is labels
    """
    np.random.seed(random_state)
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, y


def train_multiple_networks(X, y, num_networks=10, epochs=100, learning_rate=0.1, 
                           batch_size=32, random_seed=42):
    """Train multiple networks with different random initializations.
    
    Args:
        X: Input features
        y: Labels
        num_networks: Number of networks to train (default: 10)
        epochs: Number of training epochs per network (default: 100)
        learning_rate: Learning rate for SGD optimizer (default: 0.1)
        batch_size: Batch size for training (default: 32)
        random_seed: Base random seed (default: 42)
        
    Returns:
        tuple: (all_loss_histories, all_weight_trajectories)
            - all_loss_histories: List of loss histories for each network
            - all_weight_trajectories: List of weight trajectories for each network
    """
    # Create dataloader (no shuffling to ensure same order for all networks)
    dataloader = create_dataloader(X, y, batch_size=batch_size, shuffle=False)
    
    all_loss_histories = []
    all_weight_trajectories = []
    
    print(f"Training {num_networks} networks...")
    
    for i in range(num_networks):
        # Set random seed for reproducible but different initializations
        torch.manual_seed(random_seed + i)
        np.random.seed(random_seed + i)
        
        # Create and train network
        model = SimpleNN()
        loss_history, weight_trajectory = train_network(
            model, dataloader, epochs=epochs, lr=learning_rate
        )
        
        all_loss_histories.append(loss_history)
        all_weight_trajectories.append(weight_trajectory)
        
        final_loss = loss_history[-1]
        print(f"Network {i+1}/{num_networks} - Final loss: {final_loss:.4f}")
    
    return all_loss_histories, all_weight_trajectories


def apply_pca_to_trajectories(all_weight_trajectories, n_components=2):
    """Apply PCA to reduce weight trajectories to lower dimensions.
    
    Args:
        all_weight_trajectories: List of weight trajectories (each is a list of weight vectors)
        n_components: Number of PCA components (default: 2)
        
    Returns:
        tuple: (pca, all_trajectories_pca)
            - pca: Fitted PCA object
            - all_trajectories_pca: List of transformed trajectories
    """
    # Flatten all weight vectors into a single array for PCA fitting
    all_weights = []
    for trajectory in all_weight_trajectories:
        all_weights.extend(trajectory)
    all_weights = np.array(all_weights)
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    pca.fit(all_weights)
    
    # Transform each trajectory
    all_trajectories_pca = []
    for trajectory in all_weight_trajectories:
        trajectory_array = np.array(trajectory)
        trajectory_pca = pca.transform(trajectory_array)
        all_trajectories_pca.append(trajectory_pca)
    
    explained_variance = pca.explained_variance_ratio_
    print(f"\nPCA explained variance ratio: {explained_variance}")
    print(f"Total variance explained: {np.sum(explained_variance):.4f}")
    
    return pca, all_trajectories_pca


def visualize_loss_landscape(all_trajectories_pca, all_loss_histories, 
                            figsize=(14, 6)):
    """Visualize the loss landscape and training trajectories.
    
    Args:
        all_trajectories_pca: List of PCA-transformed weight trajectories
        all_loss_histories: List of loss histories for each network
        figsize: Figure size (default: (14, 6))
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Set style
    sns.set_style("whitegrid")
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_trajectories_pca)))
    
    # Plot 1: Weight trajectories in PCA space
    ax1 = axes[0]
    for i, trajectory_pca in enumerate(all_trajectories_pca):
        # Plot trajectory
        ax1.plot(trajectory_pca[:, 0], trajectory_pca[:, 1], 
                alpha=0.6, linewidth=2, color=colors[i], label=f'Network {i+1}')
        
        # Mark start point
        ax1.scatter(trajectory_pca[0, 0], trajectory_pca[0, 1], 
                   s=100, marker='o', color=colors[i], edgecolors='black', 
                   linewidths=2, zorder=5)
        
        # Mark end point
        ax1.scatter(trajectory_pca[-1, 0], trajectory_pca[-1, 1], 
                   s=100, marker='*', color=colors[i], edgecolors='black', 
                   linewidths=2, zorder=5)
    
    ax1.set_xlabel('First Principal Component', fontsize=12)
    ax1.set_ylabel('Second Principal Component', fontsize=12)
    ax1.set_title('Weight Trajectories in PCA Space', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss curves over epochs
    ax2 = axes[1]
    for i, loss_history in enumerate(all_loss_histories):
        epochs = range(len(loss_history))
        ax2.plot(epochs, loss_history, alpha=0.7, linewidth=2, 
                color=colors[i], label=f'Network {i+1}')
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training Loss Curves', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    """Main function to run the loss landscape visualization."""
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate dataset
    print("Generating dataset...")
    X, y = generate_dataset(n_samples=200, noise=0.2, random_state=42)
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    
    # Training parameters
    num_networks = 10
    epochs = 100
    learning_rate = 0.1
    batch_size = 32
    
    # Train multiple networks
    all_loss_histories, all_weight_trajectories = train_multiple_networks(
        X, y, 
        num_networks=num_networks, 
        epochs=epochs, 
        learning_rate=learning_rate,
        batch_size=batch_size,
        random_seed=42
    )
    
    # Apply PCA to weight trajectories
    print("\nApplying PCA to weight trajectories...")
    pca, all_trajectories_pca = apply_pca_to_trajectories(
        all_weight_trajectories, n_components=2
    )
    
    # Visualize results
    print("\nGenerating visualization...")
    fig = visualize_loss_landscape(all_trajectories_pca, all_loss_histories)
    plt.savefig('loss_landscape_visualization.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'loss_landscape_visualization.png'")
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    final_losses = [history[-1] for history in all_loss_histories]
    print(f"Final loss - Mean: {np.mean(final_losses):.4f}, Std: {np.std(final_losses):.4f}")
    print(f"Final loss - Min: {np.min(final_losses):.4f}, Max: {np.max(final_losses):.4f}")
    
    # Calculate distances between final weight vectors in PCA space
    final_positions = np.array([traj[-1] for traj in all_trajectories_pca])
    distances = []
    for i in range(len(final_positions)):
        for j in range(i+1, len(final_positions)):
            dist = np.linalg.norm(final_positions[i] - final_positions[j])
            distances.append(dist)
    
    print(f"\nDistance between final positions in PCA space:")
    print(f"Mean: {np.mean(distances):.4f}, Std: {np.std(distances):.4f}")
    print(f"Min: {np.min(distances):.4f}, Max: {np.max(distances):.4f}")


if __name__ == "__main__":
    main()
