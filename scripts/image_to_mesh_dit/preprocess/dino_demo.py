# I built this script with information from the following sources:
# https://github.com/MartinBurian/dinov2/blob/experiments/experiments/fg_segmantation.ipynb?short_path=ccd70b7
# https://github.com/facebookresearch/dinov2/issues/23
# https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md
# Model Details
# The model takes an image as input and returns a class token and patch tokens, and optionally 4 register tokens.
#
# The embedding dimension is:
#
# 384 for ViT-S.
# 768 for ViT-B.
# 1024 for ViT-L.
# 1536 for ViT-g.
# The models follow a Transformer architecture, with a patch size of 14. In the case of registers, we add 4 register 
# tokens, learned during training, to the input sequence after the patch embedding.
#
# For a 224x224 image, this results in 1 class token + 256 patch tokens, and optionally 4 register tokens.
#
# The models can accept larger images provided the image shapes are multiples of the patch size (14). If this 
# condition is not verified, the model will crop to the closest smaller multiple of the patch size.
import torch
import numpy as np
import cv2
import torchvision.transforms as tt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
import os
import random

# ===========================
#  MODEL SELECTION & CONFIG
# ===========================
model_name = 'dinov2_vits14_reg'
device = 'cpu'

feat_dim = {
    'dinov2_vits14_reg': 384,
    'dinov2_vits14': 384,
    'dinov2_vitb14_reg': 768,
    'dinov2_vitb14': 768,
    'dinov2_vitl14_reg': 1024,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14_reg': 1536,
    'dinov2_vitg14': 1536,
}[model_name]

mask_multiplier = {
    'dinov2_vits14_reg': 0.5,
    'dinov2_vits14': 0.5,
    'dinov2_vitb14_reg': 0.4,
    'dinov2_vitb14': 0.4,
    'dinov2_vitl14_reg': 0.4,
    'dinov2_vitl14': 0.35,
    'dinov2_vitg14_reg': 0.4,
    'dinov2_vitg14': 0.3
}[model_name]

inverse_mask = {
    'dinov2_vits14_reg': True,
    'dinov2_vits14': False,
    'dinov2_vitb14_reg': False,
    'dinov2_vitb14':False,
    'dinov2_vitl14_reg': True,
    'dinov2_vitl14': False,
    'dinov2_vitg14_reg': True,
    'dinov2_vitg14': False
}[model_name]

# ===========================
#  LOAD DINOv2 MODEL
# ===========================

def load_model(model_name):
    print(f"Loading model: {model_name}...")
    dinov2_model = torch.hub.load('facebookresearch/dinov2', model_name).to(device)
    model = dinov2_model.eval()
    print("Model loaded successfully.")
    return model


def get_random_images(folder_path, num_images=4):
    all_images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(".png")]
    if len(all_images) < num_images:
        raise ValueError(f"Not enough PNG files in the folder {folder_path}. Found {len(all_images)}, need {num_images}.")
    return random.sample(all_images, num_images)


def preprocess_images(image_paths, img_size=(448, 448)):
    images = []
    for path in image_paths:
        image = cv2.imread(path)
        image = cv2.resize(image, img_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype('float32') / 255
        images.append(image)
    images_arr = np.stack(images)
    input_tensor = torch.Tensor(np.transpose(images_arr, [0, 3, 2, 1]))
    transform = tt.Compose([tt.Normalize(mean=0.5, std=0.2)])
    return transform(input_tensor), images, image_paths

def extract_features(model, input_tensor, image_paths):
    features = model.forward_features(input_tensor)['x_norm_patchtokens'].detach()
    
    # Save features for each image
    for i, path in enumerate(image_paths):
        filename = os.path.splitext(path)[0] + '_features.pt'
        torch.save(features[i], filename)
        print(f"Saved features for {path} as {filename}")
    
    return features.numpy().reshape([input_tensor.shape[0], 1024, -1])


def segment_foreground(patch_tokens):
    fg_pca = PCA(n_components=1)
    batch_size, num_patches, channels = patch_tokens.shape
    all_patches = patch_tokens.reshape(-1, channels)
    reduced_patches = fg_pca.fit_transform(all_patches)
    norm_patches = minmax_scale(reduced_patches)
    return fg_pca, norm_patches.reshape(batch_size, num_patches)


def visualize_features(images, patch_tokens, masks, object_pca, image_paths, inverse_mask):
    plt.figure(figsize=(12, 6))  # Spread columns wider and push rows closer
    plt.suptitle(f"DINOv2 Feature Visualization - {model_name}", fontsize=14)
    for i, (image, path) in enumerate(zip(images, image_paths)):
        filename = os.path.splitext(os.path.basename(path))[0]  # Remove file extension
        patch_image = np.zeros((1024, 3), dtype='float32')
        foreground_mask = masks[i] if inverse_mask else ~masks[i]
        transformed_patches = object_pca.transform(patch_tokens[i, foreground_mask, :])
        transformed_patches = minmax_scale(transformed_patches)
        patch_image[foreground_mask, :] = transformed_patches
        color_patches = patch_image.reshape([32, -1, 3]).transpose([1, 0, 2])
        
        plt.subplot(2, 4, i + 1)
        plt.imshow(image)
        plt.title(filename, fontsize=10)
        plt.axis('off')
        
        plt.subplot(2, 4, i + 5)
        plt.imshow(color_patches)
        plt.title(filename, fontsize=10)
        plt.axis('off')
    plt.tight_layout(pad=0.8)  # Reduce padding further to bring rows closer
    plt.show()


def main():
    torch.device("cpu")
    model = load_model(model_name)
    folder_path = 'D://test/render/vvMix-98/256_vvMix-98/'  # Set folder path
    image_paths = get_random_images(folder_path, 4)
    input_tensor, images, image_paths = preprocess_images(image_paths)
    patch_tokens = extract_features(model, input_tensor,image_paths)
    fg_pca, image_norm_patches = segment_foreground(patch_tokens)
    masks = [(p > mask_multiplier).ravel() for p in image_norm_patches] 
    object_pca = PCA(n_components=3)
    fg_patches = np.vstack([patch_tokens[i, masks[i], :] for i in range(4)])
    object_pca.fit(fg_patches)
    visualize_features(images, patch_tokens, masks, object_pca, image_paths, inverse_mask)

if __name__ == "__main__":
    main()