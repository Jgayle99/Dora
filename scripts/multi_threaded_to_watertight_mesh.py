#  Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# This is a modified script from the original: Dora team (https://github.com/Seed3D/Dora)
#
# Changes:
#   Added a multi-threaded processing capability to the original script.
#   Added a progress bar with ETA using tqdm.
#   Added cli argument parsing for input folder, output folder, resolution, and number of workers.
#   Added handling of multiple file formats (glb, stl, obj).
#   Removed The JSON file path loading in favor of loading files from a folder.
# # -------------------------------------------------------------------------------

"""
Example CLI command:
    python multi_threaded_to_watertight_mesh.py --input_folder /path/to/input/folder \
        --output_folder /path/to/output/folder --resolution 512 --workers 4

Global Parameters:
    - input_folder: Path to the folder containing input mesh files (glb, stl, or obj).
    - output_folder: Path to the folder where remeshed output OBJ files will be saved.
    - resolution: Target resolution for remeshing (default: 512).
    - workers: Number of worker threads to use (default: 4).

This script preserves the original functional pipeline for remeshing as the script to_watertight_mesh.py:
  1. Load the mesh and normalize it to fit in [-1,1]^3.
  2. Build a GPU-based BVH using cubvh.
  3. Compute an unsigned distance field over a dense grid.
  4. Use DiffDMC to extract an isosurface at eps = 2/resolution.
  5. Map vertices back to the original bounding box.
  6. Split the mesh into connected components and keep the largest one.
  7. Export the result as an OBJ file.
"""

import cubvh
import torch
import numpy as np
import trimesh
from diso import DiffDMC
import argparse
from tqdm import tqdm
import os
import concurrent.futures
import threading
from datetime import datetime

def timestamp():
    """
    Returns the current timestamp as a formatted string (YYYY-MM-DD HH:MM:SS).
    Used to prepend each log message with a timestamp.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# -------------------------------------------------------------------------------
# Function: generate_dense_grid_points
# Description:
#   Generates a dense grid of 3D points covering a bounding box defined by bbox_min and
#   bbox_max. This grid is later used to sample the unsigned distance field from the mesh.
# Parameters:
#   - bbox_min: Minimum coordinates of the bounding box.
#   - bbox_max: Maximum coordinates of the bounding box.
#   - resolution: The number of intervals along each axis.
#   - indexing: Order for meshgrid (default: "ij").
# Returns:
#   - xyz: An (N, 3) array of grid points.
#   - grid_size: A list representing the grid dimensions.
# -------------------------------------------------------------------------------
def generate_dense_grid_points(
    bbox_min = np.array((-1.05, -1.05, -1.05)),
    bbox_max = np.array((1.05, 1.05, 1.05)),
    resolution = 512,
    indexing = "ij"
):
    # Create linearly spaced points along each axis.
    x = np.linspace(bbox_min[0], bbox_max[0], resolution + 1, dtype=np.float32)
    y = np.linspace(bbox_min[1], bbox_max[1], resolution + 1, dtype=np.float32)
    z = np.linspace(bbox_min[2], bbox_max[2], resolution + 1, dtype=np.float32)
    # Create a 3D grid of points.
    xs, ys, zs = np.meshgrid(x, y, z, indexing=indexing)
    # Stack into a (N, 3) array.
    xyz = np.stack((xs, ys, zs), axis=-1).reshape(-1, 3)
    # Grid size is resolution+1 along each axis.
    grid_size = [resolution + 1, resolution + 1, resolution + 1]
    return xyz, grid_size

# -------------------------------------------------------------------------------
# Function: remesh
# Description:
#   Remeshes a given input mesh using a dense grid. The process is as follows:
#     1. Load the mesh and normalize it to fit within [-1,1]^3.
#     2. Build a BVH using cubvh on the GPU.
#     3. Compute the unsigned distance field (UDF) for the grid points.
#     4. Use DiffDMC to extract an isosurface at eps = 2/resolution.
#     5. Remap the extracted vertices back to the original scale.
#     6. Extract the largest connected component.
#     7. Export the remeshed mesh as an OBJ file.
# This function is functionally identical to your original implementation.
# Parameters:
#   - grid_xyz: Dense grid of points (as a CUDA tensor).
#   - grid_size: Dimensions of the grid.
#   - mesh_path: Input mesh file path.
#   - remesh_path: Output OBJ file path.
#   - resolution: Target resolution for remeshing.
# -------------------------------------------------------------------------------
def remesh(grid_xyz, grid_size, mesh_path, remesh_path, resolution):
    worker_id = threading.current_thread().name
    tqdm.write(f"[{timestamp()}][{worker_id}][remesh] Started processing: {mesh_path} at resolution {resolution}")
    eps = 2 / resolution
    mesh = trimesh.load(mesh_path, force='mesh')
    tqdm.write(f"[{timestamp()}][{worker_id}][remesh] Mesh loaded.")

    # Normalize mesh to [-1,1]
    vertices = mesh.vertices
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) / 2
    scale = 2.0 / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale
    tqdm.write(f"[{timestamp()}][{worker_id}][remesh] Mesh normalized.")

    # Build BVH with cubvh on GPU.
    f = cubvh.cuBVH(torch.as_tensor(vertices, dtype=torch.float32, device='cuda'),
                     torch.as_tensor(mesh.faces, dtype=torch.float32, device='cuda'))
    tqdm.write(f"[{timestamp()}][{worker_id}][remesh] BVH built on GPU.")

    # Compute unsigned distance field over the grid.
    grid_udf, _ ,_= f.unsigned_distance(grid_xyz, return_uvw=False)
    tqdm.write(f"[{timestamp()}][{worker_id}][remesh] Unsigned distance computed.")

    grid_udf = grid_udf.view((grid_size[0], grid_size[1], grid_size[2]))
    diffdmc = DiffDMC(dtype=torch.float32).cuda()
    tqdm.write(f"[{timestamp()}][{worker_id}][remesh] DiffDMC instantiated.")

    # Extract the isosurface.
    vertices, faces = diffdmc(grid_udf, isovalue=eps, normalize=False)
    tqdm.write(f"[{timestamp()}][{worker_id}][remesh] DiffDMC output computed.")

    # Remap vertices to original scale.
    bbox_min = np.array((-1.05, -1.05, -1.05))
    bbox_max = np.array((1.05, 1.05, 1.05))
    bbox_size = bbox_max - bbox_min
    vertices = (vertices + 1) / grid_size[0] * bbox_size[0] + bbox_min[0]

    # Construct new mesh from extracted vertices and faces.
    mesh = trimesh.Trimesh(vertices=vertices.cpu().numpy(), faces=faces.cpu().numpy())
    tqdm.write(f"[{timestamp()}][{worker_id}][remesh] Remeshed mesh constructed.")

    # Split into connected components and keep the largest component.
    components = mesh.split(only_watertight=False)
    tqdm.write(f"[{timestamp()}][{worker_id}][remesh] Mesh split into {len(components)} components.")
    bbox_sizes = [((c.vertices.max(0) - c.vertices.min(0)).max()) for c in components]
    max_component = np.argmax(bbox_sizes)
    mesh = components[max_component]
    tqdm.write(f"[{timestamp()}][{worker_id}][remesh] Largest component extracted.")

    # Export the result as an OBJ file.
    mesh.export(remesh_path)
    tqdm.write(f"[{timestamp()}][{worker_id}][remesh] Exported remeshed file to {remesh_path}")

# -------------------------------------------------------------------------------
# Function: process_mesh_file
# Description:
#   Processes a single mesh file. It generates a dense grid for the target resolution,
#   calls the remesh function, and saves the output as an OBJ file with the target
#   resolution prefixed to the filename.
# Parameters:
#   - mesh_path: Path to the input mesh file.
#   - output_folder: Folder where the output OBJ file will be saved.
#   - target_resolution: The resolution used for remeshing.
# -------------------------------------------------------------------------------
def process_mesh_file(mesh_path, output_folder, target_resolution):
    worker_id = threading.current_thread().name
    try:
        tqdm.write(f"[{timestamp()}][{worker_id}][worker] Processing {mesh_path} at resolution {target_resolution}")
        # Generate grid for the target resolution.
        grid_xyz, grid_size = generate_dense_grid_points(resolution=target_resolution)
        grid_xyz = torch.FloatTensor(grid_xyz).cuda()

        # Construct output filename with the resolution prefix.
        basename = os.path.basename(mesh_path)
        name, _ = os.path.splitext(basename)
        remesh_name = f"{target_resolution}_{name}.obj"
        remesh_path = os.path.join(output_folder, remesh_name)
        
        # Skip file if output already exists.
        if os.path.exists(remesh_path):
            tqdm.write(f"[{timestamp()}][{worker_id}][worker] Skipping (exists): {remesh_path}")
            return
        
        # Remesh and export.
        remesh(grid_xyz, grid_size, mesh_path, remesh_path, target_resolution)
        tqdm.write(f"[{timestamp()}][{worker_id}][worker] Finished: {remesh_path}")
    except Exception as e:
        tqdm.write(f"[{timestamp()}][{worker_id}][worker] ERROR processing {mesh_path}: {e}")

# -------------------------------------------------------------------------------
# Function: main
# Description:
#   The main entry point of the script. It:
#     1. Lists all input mesh files in the specified input folder (non-recursively).
#     2. Sets up a ThreadPoolExecutor with a user-specified number of workers.
#     3. Processes each file concurrently at the target resolution.
#     4. Displays a progress bar with ETA.
# Parameters:
#   - input_folder: Folder containing the input mesh files.
#   - output_folder: Folder where remeshed OBJ files will be saved.
#   - target_resolution: The fixed resolution to use for processing each file.
#   - num_workers: Number of worker threads to use.
# -------------------------------------------------------------------------------
def main(input_folder, output_folder, target_resolution, num_workers):
    # List files in the input folder with valid extensions.
    valid_exts = ('.glb', '.stl', '.obj')
    files = [os.path.join(input_folder, f) for f in os.listdir(input_folder)
             if f.lower().endswith(valid_exts)]
    
    # Ensure the output folder exists.
    os.makedirs(output_folder, exist_ok=True)
    total_files = len(files)
    print(f"Found {total_files} files in {input_folder} to process.\n")
    
    # Set up a progress bar with ETA.
    progress_bar = tqdm(total=total_files, desc="Processing files", unit="file", dynamic_ncols=True)
    
    # Use a ThreadPoolExecutor with the specified number of workers.
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_mesh_file, mesh_path, output_folder, target_resolution): mesh_path for mesh_path in files}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # Retrieve results and catch exceptions.
            except Exception as e:
                tqdm.write(f"[{timestamp()}][main] ERROR in worker: {e}")
            progress_bar.update(1)
    progress_bar.close()

# -------------------------------------------------------------------------------
# Script entry point:
#   Parses command-line arguments and calls main().
#   Example CLI command:
#    python multi_threaded_to_watertight_mesh.py --input_folder /path/to/input/folder \
#        --output_folder /path/to/output/folder --resolution 512 --workers 4
# -------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str,
                        help="Path to the folder containing input mesh files (glb, stl, or obj).")
    parser.add_argument("--output_folder", type=str,
                        help="Path to the folder to save remeshed output as OBJ files.")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Target resolution for remeshing (default: 512).")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of worker threads (default: 4).")
    args = parser.parse_args()
    main(args.input_folder, args.output_folder, args.resolution, args.workers)
