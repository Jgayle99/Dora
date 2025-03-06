#!/usr/bin/env python3
"""
Multiprocessing wrapper for the render script that processes OBJ files.
This wrapper only queues files that do not have a marker file (i.e. a copied OBJ)
in the output folder. A progress bar (tqdm) shows the overall progress.
"""

import argparse
import multiprocessing
import subprocess
import sys
import os
import time
from tqdm import tqdm

def process_render(obj_path, input_folder, output_folder, resolution, num_views, cam_dist):
    # Normalize and build the marker file path.
    mesh_name = os.path.splitext(os.path.basename(obj_path))[0]
    marker_file = os.path.abspath(os.path.normpath(os.path.join(output_folder, mesh_name, f"{mesh_name}.obj")))
    print(f"DEBUG: Checking marker for {obj_path} => {marker_file}")
    if os.path.exists(marker_file):
        print(f"DEBUG: Skipping {obj_path}, marker exists.")
        return

    # Build the command to run the render script.
    command = [
        sys.executable,
        "D://dev/Dora-Release/Dora/pytorch_lightning/render_script.py",
        "--",  # separator so that the render script gets its arguments correctly
        "--mesh_file", obj_path,
        "--input_folder", input_folder,
        "--output_folder", output_folder,
        "--resolution", str(resolution),
        "--num_views", str(num_views),
        "--cam_dist", str(cam_dist)
    ]
    print(f"Processing: {obj_path}")
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"Process for {obj_path} exited with code {result.returncode}")

def main():
    parser = argparse.ArgumentParser(
        description="Multiprocessing wrapper for render script with progress reporting."
    )
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel worker processes (default: 4)")
    parser.add_argument("--input_folder", type=str, default="D://trainingModels/output_subsets/output_mix_5",
                        help="Folder containing input OBJ files")
    parser.add_argument("--output_folder", type=str, default="D://processed",
                        help="Base folder where render folders will be created")
    parser.add_argument("--resolution", type=int, default=1024,
                        help="Render resolution (square) (default: 1024)")
    parser.add_argument("--num_views", type=int, default=36,
                        help="Number of views to render (default: 36)")
    parser.add_argument("--cam_dist", type=float, default=2.0,
                        help="Camera distance from the object (default: 2.0)")
    args = parser.parse_args()

    # Convert folders to absolute normalized paths.
    input_folder = os.path.abspath(os.path.normpath(args.input_folder))
    output_folder = os.path.abspath(os.path.normpath(args.output_folder))

    # Gather all OBJ files from the input folder.
    all_obj_files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith('.obj')
    ]
    obj_files = []
    for obj_path in all_obj_files:
        mesh_name = os.path.splitext(os.path.basename(obj_path))[0]
        marker_file = os.path.abspath(os.path.normpath(os.path.join(output_folder, mesh_name, f"{mesh_name}.obj")))
        if not os.path.exists(marker_file):
            obj_files.append(obj_path)
        else:
            print(f"DEBUG: Skipping {obj_path}, marker exists: {marker_file}")

    total_files = len(obj_files)
    if total_files == 0:
        print("No new OBJ files found for processing.")
        return

    print(f"Total new OBJ files found: {total_files}")

    pool = multiprocessing.Pool(processes=args.workers)
    pbar = tqdm(total=total_files, desc="Processed OBJ files", unit="file")

    def update_progress(_):
        pbar.update()

    jobs = []
    for obj_path in obj_files:
        job = pool.apply_async(
            process_render,
            args=(obj_path, input_folder, output_folder, args.resolution, args.num_views, args.cam_dist),
            callback=update_progress
        )
        jobs.append(job)

    pool.close()
    while any(not job.ready() for job in jobs):
        time.sleep(1)

    pool.join()
    pbar.close()
    print("All processing complete.")

if __name__ == "__main__":
    main()
