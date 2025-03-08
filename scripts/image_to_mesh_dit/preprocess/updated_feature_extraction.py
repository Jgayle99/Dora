import torch
import os
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import concurrent.futures
from tqdm import tqdm

# ===========================
#  MODEL SELECTION & CONFIG
# ===========================
model_name = "facebook/dinov2-small"  # Small Non-Reg Model
device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
DEFAULT_WORKERS = 8  # Default to 8 workers, adjust dynamically

# ===========================
#  LOAD DINOv2 MODEL ONCE
# ===========================
def load_model_and_processor():
    """Loads the DINOv2 model and processor once."""
    print(f"Loading model: {model_name} on {device}...")
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(model_name)
    print("Model and processor loaded successfully.")
    return model, processor

def get_images(folder_path):
    """Fetch PNG images from a folder."""
    all_images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(".png")]
    return all_images if all_images else None

def is_folder_already_processed(folder_path):
    """Check if the folder already contains .pt files (indicating it was processed)."""
    return any(f.endswith(".pt") for f in os.listdir(folder_path))

def preprocess_images(image_paths, processor, device):
    """Use AutoImageProcessor to handle preprocessing and move inputs to the correct device."""
    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]  # Open images properly
    inputs = processor(images=images, return_tensors="pt")  # Convert images to tensors
    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}  # Move tensors to GPU
    return inputs, image_paths

def extract_features(model, inputs, image_paths):
    """Extract features using the model and save them."""
    with torch.no_grad():
        outputs = model(**inputs)  # Ensure model inference happens on the correct device
    
    # Extract feature embeddings
    features = outputs.last_hidden_state.detach().cpu()  # Move back to CPU before saving

    # Save features for each image
    for i, path in enumerate(image_paths):
        filename = os.path.splitext(path)[0] + '_features.pt'
        torch.save(features[i], filename)
    
    return features.numpy()

def process_folder(folder_path, model, processor, device, progress_bar):
    """Process a single folder using the shared model."""
    try:
        # Locate the actual image subfolder (e.g., D:\test\render\mesh001\256_mesh001)
        subfolders = [os.path.join(folder_path, sub) for sub in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, sub))]
        
        if not subfolders:
            print(f"No subfolders found in {folder_path}, skipping.")
            progress_bar.update(1)
            return
        
        image_folder = subfolders[0]  # Assuming there's only one subfolder per mesh

        # Check if already processed
        if is_folder_already_processed(image_folder):
            print(f"Skipping already processed folder: {image_folder}")
            progress_bar.update(1)
            return

        image_paths = get_images(image_folder)
        if not image_paths:
            print(f"No images found in {image_folder}, skipping.")
            progress_bar.update(1)
            return

        inputs, image_paths = preprocess_images(image_paths, processor, device)  # Pass device
        extract_features(model, inputs, image_paths)
        progress_bar.update(1)  # Update progress bar after processing folder
    except Exception as e:
        print(f"Error processing {folder_path}: {e}")
        progress_bar.update(1)  # Ensure progress updates even on error

def main():
    """Main function to process multiple folders concurrently using a shared model."""
    root_dir = "D:\\test\\processed"  # Root directory
    subfolders = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

    # Filter only subfolders that have not been processed
    subfolders = [folder for folder in subfolders if not is_folder_already_processed(folder)]

    if not subfolders:
        print("No new folders to process, exiting.")
        return

    num_folders = len(subfolders)
    num_workers = min(DEFAULT_WORKERS, num_folders)  # Dynamically set worker count

    # Load model and processor once in the main thread
    model, processor = load_model_and_processor()

    # Use concurrent processing with ThreadPoolExecutor to share the model across threads
    with tqdm(total=num_folders, desc="Processing Folders", unit="folder") as progress_bar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_folder, folder, model, processor, device, progress_bar): folder for folder in subfolders}
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing folder {futures[future]}: {e}")

if __name__ == "__main__":
    main()
