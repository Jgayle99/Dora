import torch
import numpy as np
import os
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# ===========================
#  MODEL SELECTION & CONFIG
# ===========================
model_name = "facebook/dinov2-small"  # Small Non-Reg Model
device = "cpu"

# ===========================
#  LOAD DINOv2 MODEL
# ===========================

def load_model(model_name, device):
    """Loads the DINOv2 model using Transformers."""
    print(f"Loading model: {model_name}...")
    try:
        model = AutoModel.from_pretrained(model_name).to(device)
        model = model.eval()
        print("Model loaded successfully.")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

def get_processor(model_name):
    """Gets the correct image processor (AutoImageProcessor will handle resizing)."""
    return AutoImageProcessor.from_pretrained(model_name)

def get_images(folder_path):
    """Fetch PNG images from a folder."""
    all_images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(".png")]
    if not all_images:
        raise ValueError(f"No PNG files found in the folder {folder_path}.")
    return all_images

def preprocess_images(image_paths, processor):
    """Use AutoImageProcessor to handle preprocessing (including resizing)."""
    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]  # Open images properly
    inputs = processor(images=images, return_tensors="pt")  # Now passing actual image objects
    return inputs, image_paths

def extract_features(model, inputs, image_paths):
    """Extract features using the model and save them."""
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract feature embeddings
    features = outputs.last_hidden_state.detach()

    # Save features for each image
    for i, path in enumerate(image_paths):
        filename = os.path.splitext(path)[0] + '_features.pt'
        torch.save(features[i], filename)
        # print(f"Saved features for {path} as {filename}")
    
    return features.numpy()

def main():
    """Main function to process images and extract features."""
    model = load_model(model_name, device)
    processor = get_processor(model_name)
    folder_path = "D://test/render/vvMix-98/256_vvMix-98/"  # Set folder path
    image_paths = get_images(folder_path)
    inputs, image_paths = preprocess_images(image_paths, processor)
    extract_features(model, inputs, image_paths)

if __name__ == "__main__":
    main()
