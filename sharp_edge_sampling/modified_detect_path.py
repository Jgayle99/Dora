#!/usr/bin/env python3
import os
import json
import argparse
import re

def find_files(directory, file_type):
    """
    Recursively finds all files in 'directory' that end with 'file_type'.
    Returns a flat list of absolute paths formatted with two forward slashes after the drive letter.
    """
    files_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(file_type):
                # Get absolute path, convert backslashes to forward slashes
                abs_path = os.path.abspath(os.path.join(root, file))
                abs_path = abs_path.replace("\\", "/")
                # Use regex to ensure the drive letter is followed by two forward slashes.
                abs_path = re.sub(r'^([A-Za-z]):/', r'\1://', abs_path)
                files_list.append(abs_path)
    return files_list

def main(directory_to_search, json_file_path, file_type):
    files = find_files(directory_to_search, file_type)
    with open(json_file_path, 'w') as f:
        json.dump(files, f, indent=4)
    print(f"Found {len(files)} {file_type} files and saved to {json_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recursively build a JSON list of absolute file paths for files matching a given type."
    )
    parser.add_argument(
        "--directory_to_search",
        type=str,
        required=True,
        help="The directory to traverse."
    )
    parser.add_argument(
        "--json_file_path",
        type=str,
        required=True,
        help="The path where the JSON file will be saved."
    )
    parser.add_argument(
        "--file_type",
        type=str,
        required=True,
        help="The file suffix to detect (e.g., .glb, .obj, .exr)."
    )
    args = parser.parse_args()
    main(args.directory_to_search, args.json_file_path, args.file_type)
