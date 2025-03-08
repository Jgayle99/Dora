import os
import py7zr
import concurrent.futures
from tqdm import tqdm

DEFAULT_WORKERS = 8  # Default number of worker threads

def archive_and_delete_mesh_and_points(subfolder, progress_bar):
    """
    Creates a 7z archive containing only the specified files in the subfolder and deletes the originals.

    Args:
        subfolder (str): The directory where the files are located.
        progress_bar (tqdm): The progress bar object for tracking.
    """
    try:
        # Get the exact subfolder name (e.g., "256_vvMix-99")
        subfolder_name = os.path.basename(subfolder)
        archive_path = os.path.join(subfolder, f"{subfolder_name}_mesh_and_points.7z")

        # Define the specific file suffixes to look for
        file_suffixes = ["_normalized.obj", "_sharp_sample.ply", "_sample_points.npz"]

        # Find all matching files in the subfolder
        matching_files = [
            os.path.join(subfolder, f)
            for f in os.listdir(subfolder)
            if any(f.endswith(suffix) for suffix in file_suffixes)
        ]

        if not matching_files:
            progress_bar.update(1)
            return  # Skip folders without matching files

        # Create a 7z archive with max compression settings
        with py7zr.SevenZipFile(archive_path, 'w', filters=[{"id": py7zr.FILTER_LZMA2, "preset": 9}]) as archive:
            for file_path in matching_files:
                archive.write(file_path, os.path.basename(file_path))  # Store with relative path

        # If archiving is successful, delete original files
        for file_path in matching_files:
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

        progress_bar.update(1)
    except Exception as e:
        print(f"Error processing {subfolder}: {e}")
        progress_bar.update(1)

def main():
    """Main function to find, archive, and delete specific files in all sub-subfolders concurrently."""
    root_dir = input("Enter the root directory: ").strip()

    if not os.path.exists(root_dir):
        print(f"Error: The directory '{root_dir}' does not exist.")
        return

    # Find all direct subfolders (e.g., "vvMix-99", "vvMix-98")
    parent_folders = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

    # Find one level deeper subfolders (e.g., "vvMix-99/256_vvMix-99")
    subfolders_to_process = []
    for parent_folder in parent_folders:
        subfolders = [os.path.join(parent_folder, sub) for sub in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, sub))]
        if subfolders:
            subfolders_to_process.append(subfolders[0])  # Assume only one subfolder per parent

    if not subfolders_to_process:
        print("No subfolders found at the required depth, exiting.")
        return

    num_folders = len(subfolders_to_process)
    num_workers = min(DEFAULT_WORKERS, num_folders)  # Dynamically set worker count

    print(f"Archiving and deleting sample files in {num_folders} deep-level folders using 7z (max compression)...")

    # Multi-threaded execution with progress tracking
    with tqdm(total=num_folders, desc="Processing Folders", unit="folder") as progress_bar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(archive_and_delete_mesh_and_points, sub, progress_bar): sub
                for sub in subfolders_to_process
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing folder {futures[future]}: {e}")

    print(f"\nSuccessfully processed {num_folders} folders.")

if __name__ == "__main__":
    main()
