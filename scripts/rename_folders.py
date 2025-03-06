#!/usr/bin/env python3
import os
import argparse
import logging
from tqdm import tqdm
import shutil

def main():
    parser = argparse.ArgumentParser(
        description="Rename subfolders by removing 'p_' prefix and moving them into a new folder based on their name."
    )
    parser.add_argument("--parent_folder", type=str, default="D://processed",
                        help="The parent folder containing subfolders to be renamed (e.g., D:\\processed).")
    args = parser.parse_args()

    parent = args.parent_folder
    # List all subfolders in the parent folder.
    subfolders = [f for f in os.listdir(parent) if os.path.isdir(os.path.join(parent, f))]
    # Filter for subfolders starting with "p_"
    subfolders = [f for f in subfolders if f.startswith("p_")]
    logging.info(f"Found {len(subfolders)} subfolders starting with 'p_' in {parent}")

    for folder in tqdm(subfolders, desc="Processing subfolders"):
        old_path = os.path.join(parent, folder)
        # Remove prefix "p_"
        new_name = folder[2:]
        # Find the first underscore in the new name.
        underscore_index = new_name.find("_")
        if underscore_index == -1:
            logging.warning(f"Skipping folder {folder}: no underscore found after prefix removal.")
            continue
        # Use the part after the first underscore as the new parent folder name.
        new_parent_folder_name = new_name[underscore_index+1:]
        new_parent_path = os.path.join(parent, new_parent_folder_name)
        # Create the new parent folder if it doesn't exist.
        os.makedirs(new_parent_path, exist_ok=True)
        new_folder_path = os.path.join(new_parent_path, new_name)
        try:
            shutil.move(old_path, new_folder_path)
            logging.info(f"Moved '{old_path}' to '{new_folder_path}'")
        except Exception as e:
            logging.error(f"Error moving '{old_path}' to '{new_folder_path}': {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
