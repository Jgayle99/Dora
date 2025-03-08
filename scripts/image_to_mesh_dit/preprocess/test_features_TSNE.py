import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse

def visualize_features_tsne(feature_path, save_path=None):
    """Loads and visualizes feature extraction using t-SNE"""
    
    # Load feature file
    features = torch.load(feature_path, map_location='cpu')
    
    # Ensure the tensor shape is correct
    if len(features.shape) == 3:  # (batch, num_patches, feature_dim)
        features = features.squeeze(0)  

    print(f"Feature shape: {features.shape}")

    # Convert tensor to NumPy (convert BFloat16 to Float32 first)
    features = features.to(torch.float32).numpy()

    # Apply t-SNE to reduce dimensionality to 2D
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, learning_rate=200.0, n_iter=1000)
    features_reduced = tsne.fit_transform(features)  # Map to 2D

    # Normalize for better visualization
    features_reduced = (features_reduced - features_reduced.min()) / (features_reduced.max() - features_reduced.min())

    # Plot t-SNE visualization
    plt.figure(figsize=(8, 8))
    plt.scatter(features_reduced[:, 0], features_reduced[:, 1], c=features_reduced[:, 1], cmap='viridis', alpha=0.75)
    plt.colorbar(label="Feature Value")
    plt.title("Feature Visualization with t-SNE")
    plt.axis("off")

    # Save the image if requested
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved feature visualization to {save_path}")

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize DINOv2 extracted features using t-SNE.")
    parser.add_argument("--feature_path", type=str, default="D://test/render/vvMix-98/256_vvMix-98/p_256_vvMix-98_view_000_depth_features.pt", help="Path to the .pt feature file")
    parser.add_argument("--save_path", type=str, default=None, help="Optional path to save the visualization")
    args = parser.parse_args()

    visualize_features_tsne(args.feature_path, args.save_path)
