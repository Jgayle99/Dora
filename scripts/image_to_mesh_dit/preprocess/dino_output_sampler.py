import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
import os
import random
from PIL import Image

# ===========================
#  MODEL SELECTION & CONFIG
# ===========================
model_name = 'dinov2_vits14'  # Set your model
device = 'cpu'

# Define embedding dimensions for each model variant
feat_dim = {
    'dinov2_vits14_reg': 384,
    'dinov2_vits14': 384,
    'dinov2_vitb14_reg': 768,
    'dinov2_vitb14': 768,
    'dinov2_vitl14_reg': 1024,  # Large model
    'dinov2_vitl14': 1024,  # Large model
    'dinov2_vitg14_reg': 1536,
    'dinov2_vitg14': 1536,
}[model_name]

IMAGE_SIZE = 224  # We changed it to 448x448
PATCH_SIZE = 14
PATCH_GRID_SIZE = IMAGE_SIZE // PATCH_SIZE  # 32 for 448x448 images
num_files = 4  # Number of .pt files to randomly select

def get_random_pt_files(folder_path, num_files=4):
    all_files = [f for f in os.listdir(folder_path) if f.lower().endswith("_features.npz")]
    if len(all_files) < num_files:
        raise ValueError(f"Not enough _features.npz files in the folder {folder_path}. Found {len(all_files)}, need {num_files}.")
    return random.sample(all_files, num_files)

def load_features(folder_path, file_names):
    feature_data = []
    for name in file_names:
        data = np.load(os.path.join(folder_path, name))  # Load each npz file
        features = data['features']  # Extract stored features
        feature_data.append(features)  # Append separately for each file
        print(f"Loaded {name}, shape: {features.shape}, min: {features.min()}, max: {features.max()}")
    return feature_data

def process_features(feature_data):
    object_pca = PCA(n_components=3)
    feature_matrices = []

    for features in feature_data:
        reshaped_features = features.reshape(PATCH_GRID_SIZE ** 2, feat_dim)

        # Perform PCA transformation
        transformed_patches = object_pca.fit_transform(reshaped_features)
        transformed_patches = minmax_scale(transformed_patches)

        # Reshape back to expected visualization format (32x32)
        feature_matrices.append(transformed_patches.reshape(PATCH_GRID_SIZE, PATCH_GRID_SIZE, 3))

    return object_pca, feature_matrices

def visualize_features(folder_path, file_names, feature_matrices):
    plt.figure(figsize=(12, 6))  # Adjust figure size
    plt.suptitle(f"Feature Extraction Visualization - {model_name}", fontsize=14)

    for i, name in enumerate(file_names):
        base_name = name.replace("_features.npz", "")  # Remove _features.npz suffix
        png_path = os.path.join(folder_path, base_name + ".png")

        # Load and display the PNG image (if available)
        plt.subplot(2, num_files, i + 1)
        if os.path.exists(png_path):
            img = Image.open(png_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))  # Resize to match 448x448
            plt.imshow(img)
        else:
            plt.text(0.5, 0.5, "No Image", fontsize=10, ha='center', va='center')
        plt.axis('off')

        # Processed feature visualization
        feature_matrix = feature_matrices[i]

        # Resize for display
        feature_image = np.array(Image.fromarray((feature_matrix * 255).astype(np.uint8)).resize((IMAGE_SIZE, IMAGE_SIZE)))

        plt.subplot(2, num_files, i + num_files + 1)
        plt.imshow(feature_image)
        plt.title(base_name, fontsize=10)
        plt.axis('off')

    plt.tight_layout(pad=0.8)
    plt.show()

def main():
    folder_path = "D://test/render/vvMix-99/256_vvMix-99"  # Set your folder path
    file_names = get_random_pt_files(folder_path, num_files)
    feature_data = load_features(folder_path, file_names)
    object_pca, feature_matrices = process_features(feature_data)
    visualize_features(folder_path, file_names, feature_matrices)

if __name__ == "__main__":
    main()
