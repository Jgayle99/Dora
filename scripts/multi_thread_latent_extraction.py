#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import numpy as np
import torch
import pytorch_lightning as pl
import time
import datetime
import shutil

# Import configuration utilities from craftsman
from craftsman.utils.config import load_config
from craftsman.utils.misc import get_rank
import craftsman  # ensure craftsman is imported

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def process_npz_file(npz_path, vae, device, output_root):
    """
    Process a single NPZ file to extract latent code using Dora-VAE.
    Creates an output subfolder (named as the NPZ file without extension)
    and saves the latent code there with the suffix "_latent_sample.npy".
    Skips processing if the output file already exists.
    """
    base_name = os.path.splitext(os.path.basename(npz_path))[0]
    output_subfolder = os.path.join(output_root, base_name)
    os.makedirs(output_subfolder, exist_ok=True)
    out_path = os.path.join(output_subfolder, base_name + "_latent_sample.npy")
    if os.path.exists(out_path):
        logging.info(f"Skipping {npz_path} (output already exists).")
        return

    try:
        data = np.load(npz_path)
    except Exception as e:
        logging.error(f"Error loading NPZ file {npz_path}: {e}")
        return

    coarse = data.get("fps_coarse_surface")
    sharp = data.get("fps_sharp_surface")
    if coarse is None or sharp is None:
        logging.error(f"NPZ file {npz_path} must contain both 'fps_coarse_surface' and 'fps_sharp_surface'.")
        return

    # Squeeze extra singleton dimensions if necessary.
    if coarse.ndim == 3 and coarse.shape[1] == 1:
        coarse = np.squeeze(coarse, axis=1)
    if sharp.ndim == 3 and sharp.shape[1] == 1:
        sharp = np.squeeze(sharp, axis=1)

    # Verify that both arrays have 6 channels.
    if coarse.shape[1] != 6 or sharp.shape[1] != 6:
        logging.error(f"Expected NPZ file {npz_path} to have point arrays with 6 channels, but got shapes {coarse.shape} and {sharp.shape}.")
        return

    logging.info(f"Processing NPZ file: {npz_path} (coarse: {coarse.shape}, sharp: {sharp.shape})")

    # Convert to torch tensors with a batch dimension.
    coarse_tensor = torch.from_numpy(coarse).float().unsqueeze(0).to(device)
    sharp_tensor  = torch.from_numpy(sharp).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        start_time = time.time()
        latent_tuple = vae.encode(coarse_tensor, sharp_tensor)
        # Unpack the tuple; we use only the shape latents.
        shape_latents, kl_embed, posterior = latent_tuple
    latent = shape_latents.squeeze(0).cpu().numpy()
    np.save(out_path, latent)
    end_time = time.time()
    logging.info(f"Saved latent for {npz_path} to {out_path} in {end_time - start_time:.4f} seconds.")

def main(args, extras):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    cfg = load_config(args.config, cli_args=extras)
    device = args.device
    pl.seed_everything(cfg.seed + get_rank() + 1, workers=True)
    
    # Instantiate the system and load the shape_model (Dora-VAE)
    dm = craftsman.find(cfg.data_type)(cfg.data)
    system = craftsman.find(cfg.system_type)(cfg.system, resumed=cfg.resume is not None)
    system.to(device)
    system.eval()
    
    try:
        vae = system.shape_model
    except AttributeError:
        logging.error("The instantiated system does not have a 'shape_model' attribute.")
        sys.exit(1)
    
    if not hasattr(vae, "split"):
        vae.split = "val"
    
    # Get list of NPZ files in the input folder.
    npz_files = [os.path.join(args.input_folder, f) for f in os.listdir(args.input_folder) if f.endswith(".npz")]
    if not npz_files:
        logging.error("No NPZ files found in input folder.")
        return
    
    global_start_time = time.time()
    logging.info(f"Script started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    num_workers = min(args.num_workers, len(npz_files))
    logging.info(f"Processing {len(npz_files)} NPZ files using {num_workers} worker(s)...")
    
    # Process NPZ files concurrently.
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_npz_file, npz_path, vae, device, args.output_folder): npz_path 
                   for npz_path in npz_files}
        for _ in tqdm(as_completed(futures), total=len(futures), desc="NPZ Files Processed"):
            pass

    global_end_time = time.time()
    logging.info(f"Script ended at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Total runtime: {global_end_time - global_start_time:.4f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test latent extraction with Dora-VAE.")
    parser.add_argument("--config", type=str, default="./configs/shape-autoencoder/Dora-VAE-test.yaml",
                        help="Path to the YAML config file (e.g., Dora-VAE-test.yaml)")
    parser.add_argument("--input_folder", type=str, default="D://data/output/sample",
                        help="Folder containing NPZ files (all in one folder)")
    parser.add_argument("--output_folder", type=str, default="D://processed",
                        help="Folder to save the extracted latent codes (each in a subfolder named after the NPZ file)")
    parser.add_argument("--model_path", type=str, default="D://dev/Dora-Release/Dora/pytorch_lightning/ckpts/Dora-VAE-1.1/dora_vae_1_1.ckpt",
                        help="Path to pre-trained Dora-VAE checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker threads")
    args, extras = parser.parse_known_args()
    main(args, extras)
