#!/usr/bin/env python3
"""
This script uses Python's multiprocessing module to launch several processes,
each of which calls the 'modified_sharp_sample.py' script.
It also displays a basic progress bar (using tqdm) showing:
  - The total number of input OBJ files (based on the initial unclaimed files).
  - The number processed (as total - unclaimed).
  - Elapsed time.

Run this wrapper to parallelize processing.
"""

import argparse
import multiprocessing
import subprocess
import sys
import os
import time
from tqdm import tqdm

def run_sharp_sample(args_list):
    """
    Run the modified_sharp_sample.py script as a subprocess with the provided arguments.
    """
    # Build the command list. We assume that sys.executable will run the script.
    cmd = [sys.executable, "modified_sharp_sample.py"] + args_list
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Process exited with code {result.returncode}")

def main():
    parser = argparse.ArgumentParser(
        description="Run modified_sharp_sample.py in parallel using multiprocessing with progress reporting."
    )
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of worker processes to spawn (default: 4).")
    parser.add_argument("--input_folder", type=str, default="D://training/input",
                        help="Path to the folder containing input OBJ files (default: D://training/input).")
    parser.add_argument("--angle_threshold", type=int, default=15,
                        help="Sharpness threshold angle in degrees (default: 15).")
    parser.add_argument("--point_number", type=int, default=65536,
                        help="Number of points to sample (default: 65536).")
    parser.add_argument("--sharp_point_path", type=str, default="D://training/output/sharp_point_ply",
                        help="Directory to save the sharp point output (default: D://training/output/sharp_point_ply).")
    parser.add_argument("--sample_path", type=str, default="D://training/output/sample",
                        help="Directory to save the sample data (default: D://training/output/sample).")
    args = parser.parse_args()

    # Build the argument list to be passed to modified_sharp_sample.py.
    sharp_args = [
        "--input_folder", args.input_folder,
        "--angle_threshold", str(args.angle_threshold),
        "--point_number", str(args.point_number),
        "--sharp_point_path", args.sharp_point_path,
        "--sample_path", args.sample_path,
    ]
    
    # Count the total files (only unclaimed files, i.e. those not prefixed with "p_")
    def count_unclaimed(folder):
        return len([f for f in os.listdir(folder)
                    if f.lower().endswith('.obj') and not f.startswith("p_")])
    
    # Get the initial count (Note: if files are being claimed, the initial total is the sum of claimed and unclaimed).
    all_obj = [f for f in os.listdir(args.input_folder) if f.lower().endswith('.obj')]
    total_files = len(all_obj)
    print(f"Total OBJ files at start: {total_files}")

    # Spawn the worker processes.
    processes = []
    for i in range(args.workers):
        p = multiprocessing.Process(target=run_sharp_sample, args=(sharp_args,))
        p.start()
        print(f"Started worker {i} (PID: {p.pid})")
        processes.append(p)
    
    # Start a progress bar and timer.
    start_time = time.time()
    pbar = tqdm(total=total_files, desc="Processed files", unit="file")

    # Poll the input folder periodically.
    # Processed count = total_files - current unclaimed files.
    # Continue until all worker processes have finished.
    while any(p.is_alive() for p in processes):
        unclaimed = count_unclaimed(args.input_folder)
        processed = total_files - unclaimed
        pbar.n = processed
        elapsed = time.time() - start_time
        pbar.set_postfix({"Elapsed(sec)": f"{elapsed:.1f}", "Remaining": unclaimed})
        pbar.refresh()
        time.sleep(2)
    
    # One last update
    unclaimed = count_unclaimed(args.input_folder)
    processed = total_files - unclaimed
    pbar.n = processed
    elapsed = time.time() - start_time
    pbar.set_postfix({"Elapsed(sec)": f"{elapsed:.1f}", "Remaining": unclaimed})
    pbar.refresh()
    pbar.close()

    # Wait for all processes to finish.
    for i, p in enumerate(processes):
        p.join()
        print(f"Worker {i} (PID: {p.pid}) finished.")

if __name__ == "__main__":
    main()
