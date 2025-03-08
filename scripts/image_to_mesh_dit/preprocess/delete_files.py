import os
import sys
import concurrent.futures
from tqdm import tqdm

DEFAULT_WORKERS = 8  # Default number of workers, adjusted dynamically

def find_files_by_extension(root_dir, file_extension):
    """
    Recursively finds all files with the specified extension in the given directory.

    Args:
        root_dir (str): The root directory to start searching.
        file_extension (str): The file extension to search for (e.g., ".pt").
    
    Returns:
        list: List of file paths that match the extension.
    """
    if not os.path.exists(root_dir):
        print(f"Error: The directory '{root_dir}' does not exist.")
        return []

    files_to_delete = []
    for folder_path, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(file_extension):
                files_to_delete.append(os.path.join(folder_path, filename))
    
    return files_to_delete

def delete_file(file_path, progress_bar):
    """
    Deletes a single file and updates the progress bar.

    Args:
        file_path (str): The path of the file to delete.
        progress_bar (tqdm): The progress bar object.
    """
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")
    finally:
        progress_bar.update(1)

def delete_files_concurrently(root_dir, file_extension):
    """
    Finds and deletes files with the specified extension using multi-threading.

    Args:
        root_dir (str): The root directory to search.
        file_extension (str): The file extension to delete (e.g., ".pt").
    """
    files_to_delete = find_files_by_extension(root_dir, file_extension)

    if not files_to_delete:
        print(f"No '{file_extension}' files found in '{root_dir}'. Nothing to delete.")
        return

    # Confirmation prompt
    print(f"\nFound {len(files_to_delete)} files with '{file_extension}'.")
    confirm = input("Do you want to delete these files? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("Deletion aborted.")
        return

    # Determine the number of worker threads
    num_workers = min(DEFAULT_WORKERS, len(files_to_delete))

    # Delete files using concurrent threads with progress tracking
    with tqdm(total=len(files_to_delete), desc="Deleting Files", unit="file") as progress_bar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(delete_file, file, progress_bar): file for file in files_to_delete}
            for future in concurrent.futures.as_completed(futures):
                future.result()  # Ensure exceptions are raised if any occur

    print(f"\nSuccessfully deleted {len(files_to_delete)} '{file_extension}' files.")

if __name__ == "__main__":
    root_directory = input("Enter the root directory: ").strip()
    extension = input("Enter the file extension to delete (e.g., .pt, .txt, .png): ").strip()

    if not extension.startswith("."):
        print("Invalid extension format. Please include the dot (e.g., '.pt').")
        sys.exit(1)

    delete_files_concurrently(root_directory, extension)
