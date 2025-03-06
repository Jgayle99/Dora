#!/usr/bin/env python3
import os
import argparse
import logging
from tqdm import tqdm

def rename_file(old_path):
    """
    Rename a single file according to the rules:
      - Remove the prefix "p_" if present.
      - Remove all occurrences of "0001".
      - For .ply files, add the suffix "_sharp_sample" before the extension.
      - For .obj files, add the suffix "_normalized" before the extension.
      - For .npz files, add the suffix "_sample_points" before the extension.
    """
    dirname, filename = os.path.split(old_path)
    # Remove prefix "p_" if present
    if filename.startswith("p_"):
        filename = filename[2:]
    # Remove all occurrences of "0001"
    filename = filename.replace("0001", "")
    
    base, ext = os.path.splitext(filename)
    # Depending on the extension, add a suffix.
    if ext.lower() == ".ply":
        new_base = base + "_sharp_sample"
    elif ext.lower() == ".obj":
        new_base = base + "_normalized"
    elif ext.lower() == ".npz":
        new_base = base + "_sample_points"
    else:
        new_base = base
    new_filename = new_base + ext
    new_path = os.path.join(dirname, new_filename)
    return new_path

def main():
    parser = argparse.ArgumentParser(
        description="Rename files in subfolders by removing prefix 'p_', removing '0001', and adding extension-specific suffixes."
    )
    parser.add_argument("--input_folder", type=str, default="D://processed",
                        help="Folder containing files in subfolders to be renamed.")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Recursively collect all files in the input folder.
    files_to_rename = []
    for root, dirs, files in os.walk(args.input_folder):
        for f in files:
            files_to_rename.append(os.path.join(root, f))
    
    logging.info(f"Found {len(files_to_rename)} files to rename.")
    
    # Rename each file with a progress bar.
    for old_path in tqdm(files_to_rename, desc="Renaming files"):
        new_path = rename_file(old_path)
        # Only rename if the name has changed.
        if new_path != old_path:
            try:
                os.rename(old_path, new_path)
                logging.info(f"Renamed: {old_path} -> {new_path}")
            except Exception as e:
                logging.error(f"Error renaming {old_path} to {new_path}: {e}")

if __name__ == "__main__":
    main()
