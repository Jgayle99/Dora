#!/usr/bin/env python3
import os
import shutil
import argparse
import logging
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(
        description="Move NPZ files from an input folder into individual output subfolders (named after the NPZ file)."
    )
    parser.add_argument("--input_folder", type=str, default="D://training/output/sample",
                        help="Folder containing NPZ files.")
    parser.add_argument("--output_folder", type=str, default="D://processed",
                        help="Folder in which to create subfolders and move NPZ files.")
    args = parser.parse_args()
    
    # Create the output folder if it doesn't exist.
    os.makedirs(args.output_folder, exist_ok=True)
    
    # List all NPZ files in the input folder.
    npz_files = [f for f in os.listdir(args.input_folder) if f.endswith(".npz")]
    if not npz_files:
        logging.error("No NPZ files found in the input folder.")
        return

    logging.info(f"Found {len(npz_files)} NPZ file(s) in {args.input_folder}.")
    
    for npz in tqdm(npz_files, desc="Moving NPZ files"):
        base = os.path.splitext(npz)[0]
        output_subfolder = os.path.join(args.output_folder, base)
        os.makedirs(output_subfolder, exist_ok=True)
        
        src_path = os.path.join(args.input_folder, npz)
        dst_path = os.path.join(output_subfolder, npz)
        
        try:
            shutil.move(src_path, dst_path)
            logging.info(f"Moved {src_path} to {dst_path}")
        except Exception as e:
            logging.error(f"Error moving {src_path} to {dst_path}: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
