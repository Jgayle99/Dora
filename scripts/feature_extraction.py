import torch
import os
import argparse
import numpy as np
import time
import datetime
import shutil
from transformers import AutoConfig, AutoModel, AutoImageProcessor
from safetensors.torch import load_file  
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # for progress bars

def load_exr_image(exr_path):
    """Loads an EXR image using pyexr and returns it as a NumPy array (RGB).

    If the image is grayscale (2D) or single channel, it is replicated to form an RGB image.
    """
    try:
        import pyexr
    except ImportError:
        raise ImportError("pyexr is required to load EXR images. Please install it with 'pip install pyexr'.")
    
    try:
        image = pyexr.read(exr_path)
    except Exception as e:
        raise ValueError(f"Failed to load EXR image with pyexr: {e}")
    
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 1:
        image = np.concatenate([image] * 3, axis=-1)
    
    return image

def process_single_image(image_path, model, processor, device='cuda'):
    """Processes a single EXR image (or depth map) and extracts features silently."""
    try:
        image_np = load_exr_image(image_path)
    except Exception as e:
        return None

    # If filename contains "depth", apply min–max normalization; otherwise, scale from [-1,1] to [0,1] if needed.
    if "depth" in os.path.basename(image_path).lower():
        depth_min = image_np.min()
        depth_max = image_np.max()
        if depth_max > depth_min:
            image_np = (image_np - depth_min) / (depth_max - depth_min)
    else:
        if image_np.min() < 0 or image_np.max() > 1:
            image_np = (image_np + 1) / 2.0

    try:
        inputs = processor(images=image_np, return_tensors="pt", do_rescale=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
    except Exception as e:
        return None

    with torch.no_grad():
        if device == 'cuda':
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)
        features = outputs.last_hidden_state[:, 1:, :].to(torch.bfloat16)
    
    return features

def process_folder(folder_path, model, processor, device):
    """Processes all EXR images in a folder and saves features in the same folder.
    This function prints one line indicating which folder is being processed.
    """
    print(f"Processing folder: {folder_path}")
    
    # Use the same folder as output.
    output_folder = folder_path
    
    # List all EXR files in the folder.
    exr_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.exr')]
    if not exr_files:
        return
    
    # Process images silently with a progress bar.
    for image_path in tqdm(exr_files, desc=f"{os.path.basename(folder_path)}", leave=False):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_folder, base_name + "_features.pt")
        if os.path.exists(output_path):
            # Skip if features file already exists.
            continue
        features = process_single_image(image_path, model, processor, device)
        if features is not None:
            torch.save(features, output_path)

def main():
    global_start_time = time.time()
    print("Script started at:", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    parser = argparse.ArgumentParser(
        description="Extract DINOv2 features from multiple folders of EXR images (normal or depth) in bf16."
    )
    parser.add_argument('--exr_folder', type=str, default="D://processed", help='Path to folder containing EXR image subfolders.')
    parser.add_argument('--model_path', type=str, default="D://dev/dinov2/dinov2-with-registers-large.safetensors", help='Path to the .safetensors file.')
    parser.add_argument('--device', type=str, default='cuda', help="Device ('cuda' or 'cpu').")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of worker threads to process folders.")
    
    args = parser.parse_args()
    
    device = args.device.lower()
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA requested but not available. Using CPU.")
        device = 'cpu'
    
    # Create temporary model directory for config/processor files.
    temp_model_dir = "temp_model_dir"
    os.makedirs(temp_model_dir, exist_ok=True)
    
    safetensors_filename = os.path.basename(args.model_path)
    safetensors_dest_path = os.path.join(temp_model_dir, safetensors_filename)
    try:
        shutil.copy(args.model_path, safetensors_dest_path)
    except Exception as e:
        raise Exception(f"Error copying the model: {e}")
    
    try:
        config = AutoConfig.from_pretrained('facebook/dinov2-with-registers-large')
        config.save_pretrained(temp_model_dir)
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-with-registers-large')
        processor.save_pretrained(temp_model_dir)
    except Exception as e:
        raise Exception(f"Failed to download config or processor: {e}")
    
    print("\nLoading model and processor...")
    try:
        config = AutoConfig.from_pretrained(temp_model_dir)
        model = AutoModel.from_config(config)
        state_dict = load_file(safetensors_dest_path, device=device)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"WARNING: Missing keys in state_dict: {missing_keys}")
        if unexpected_keys:
            print(f"WARNING: Unexpected keys in state_dict: {unexpected_keys}")
    except Exception as e:
        raise Exception(f"Error loading model or processor: {e}")
    
    model.to(device)
    model.eval()
    print("Model loaded and moved to device.")
    
    # Get list of subfolders in the exr_folder.
    all_folders = [os.path.join(args.exr_folder, d) for d in os.listdir(args.exr_folder)
                   if os.path.isdir(os.path.join(args.exr_folder, d))]
    if not all_folders:
        print("No subfolders found in the specified exr_folder.")
        return
    
    num_workers = min(args.num_workers, len(all_folders))
    print(f"\nProcessing {len(all_folders)} folders using {num_workers} threads...")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_folder, folder, model, processor, device): folder 
                   for folder in all_folders}
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Folders Processed"):
            pass

    shutil.rmtree(temp_model_dir, ignore_errors=True)
    
    global_end_time = time.time()
    print("\nScript ended at:", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("Total runtime: {:.2f} seconds.".format(global_end_time - global_start_time))

if __name__ == '__main__':
    main()
