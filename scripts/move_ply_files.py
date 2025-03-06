#!/usr/bin/env python3
import os
import shutil
import argparse
import logging
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(
        description="Move ply files from an input folder into individual output subfolders (named after the ply file)."
    )
    parser.add_argument("--input_folder", type=str, default="D://training/output/sharp_point_ply",
                        help="Folder containing ply files.")
    parser.add_argument("--output_folder", type=str, default="D://processed",
                        help="Folder in which to create subfolders and move ply files.")
    args = parser.parse_args()
    
    # Create the output folder if it doesn't exist.
    os.makedirs(args.output_folder, exist_ok=True)
    
    # List all ply files in the input folder.
    ply_files = [f for f in os.listdir(args.input_folder) if f.endswith(".ply")]
    if not ply_files:
        logging.error("No ply files found in the input folder.")
        return

    logging.info(f"Found {len(ply_files)} ply file(s) in {args.input_folder}.")
    
    for ply in tqdm(ply_files, desc="Moving ply files"):
        base = os.path.splitext(ply)[0]
        output_subfolder = os.path.join(args.output_folder, base)
        os.makedirs(output_subfolder, exist_ok=True)
        
        src_path = os.path.join(args.input_folder, ply)
        dst_path = os.path.join(output_subfolder, ply)
        
        try:
            shutil.move(src_path, dst_path)
            logging.info(f"Moved {src_path} to {dst_path}")
        except Exception as e:
            logging.error(f"Error moving {src_path} to {dst_path}: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
