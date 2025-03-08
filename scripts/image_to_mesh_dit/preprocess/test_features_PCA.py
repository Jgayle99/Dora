import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse

def visualize_features_pca(feature_path, num_components=4, save_path=None):
    """Loads and visualizes feature extraction using multiple PCA components."""
    
    # Load feature file
    features = torch.load(feature_path, map_location='cpu')
    
    # Ensure the tensor shape is correct
    if len(features.shape) == 3:  # (batch, num_patches, feature_dim)
        features = features.squeeze(0)  

    print(f"Feature shape: {features.shape}")

    # Convert tensor to NumPy (convert BFloat16 to Float32 first)
    features = features.to(torch.float32).numpy()

    # Apply PCA to reduce dimensionality
    num_components = min(num_components, features.shape[1])  # Ensure we don't exceed available dimensions
    pca = PCA(n_components=num_components, random_state=42)
    features_reduced = pca.fit_transform(features)  # Reduce feature dimensions

    # Normalize each component independently for visualization
    features_reduced = (features_reduced - features_reduced.min(axis=0)) / (features_reduced.max(axis=0) - features_reduced.min(axis=0))

    # Find the closest square dimensions
    num_patches = features.shape[0]
    side_length = int(np.floor(np.sqrt(num_patches)))  # Use floor to avoid overflow
    features_reduced = features_reduced[: side_length**2]  # Crop to a perfect square

    # Create a figure for multiple PCA components
    fig, axes = plt.subplots(1, num_components, figsize=(4 * num_components, 4))
    if num_components == 1:
        axes = [axes]  # Ensure axes is iterable for single component case

    for i in range(num_components):
        feature_img = features_reduced[:, i].reshape(side_length, side_length)  # Reshape into 2D
        axes[i].imshow(feature_img, cmap='viridis')
        axes[i].axis('off')
        axes[i].set_title(f"PCA Component {i+1}")

    plt.suptitle("Feature Visualization with PCA")

    # Save the image if requested
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved feature visualization to {save_path}")

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize DINOv2 extracted features using PCA.")
    parser.add_argument("--feature_path", type=str, default="D://test/render/vvMix-98/256_vvMix-98/p_256_vvMix-98_view_000_depth_features.pt", help="Path to the .pt feature file")
    parser.add_argument("--num_components", type=int, default=4, help="Number of PCA components to visualize")
    parser.add_argument("--save_path", type=str, default=None, help="Optional path to save the visualization")
    args = parser.parse_args()

    visualize_features_pca(args.feature_path, args.num_components, args.save_path)
