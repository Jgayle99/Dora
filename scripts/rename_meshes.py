import os
import argparse

def rename_files(folder, prefix):
    # List all files in the provided folder
    files = os.listdir(folder)
    count = 1
    for filename in files:
        lower_filename = filename.lower()
        # Process only .obj and .stl files
        if lower_filename.endswith('.obj') or lower_filename.endswith('.stl'):
            # Get the original file extension
            _, ext = os.path.splitext(filename)
            new_filename = f"{prefix}-{count}{ext}"
            old_path = os.path.join(folder, filename)
            new_path = os.path.join(folder, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed '{filename}' to '{new_filename}'")
            count += 1
    print(f"Renamed {count - 1} files in total.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rename all .obj and .stl files in a folder to <prefix>-<number> while preserving the extension."
    )
    parser.add_argument("folder", help="Path to the folder containing the files.")
    parser.add_argument("prefix", help="The prefix to use for renaming the files.")
    args = parser.parse_args()
    
    rename_files(args.folder, args.prefix)
