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
# code builder: Dora team (https://github.com/Seed3D/Dora)
# This is a modified script from the original: Dora team (https://github.com/Seed3D/Dora)
#
# Changes:
#   Added a multi-processing capability to the original script.
#   Modified the cli arguments and their default values.
#   Removed The JSON file path loading in favor of loading files from a folder.
# # -------------------------------------------------------------------------------

import bpy
import os
import math
import open3d as o3d
import numpy as np
import bmesh
import argparse
import gc
import trimesh
import fpsample
from pysdf import SDF
import random

def save_vertices_as_ply_open3d(vertices, filepath):
    """
    Save the given vertices as a PLY point cloud using Open3D.
    
    Parameters:
      vertices: NumPy array of vertex positions.
      filepath: Output file path.
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(vertices)
    # Map positions from [-1,1] to [0,1] for coloring.
    point_cloud.colors = o3d.utility.Vector3dVector((vertices + 1) / 2)
    o3d.io.write_point_cloud(filepath, point_cloud, write_ascii=True)

def process_mesh(mesh_path, point_number, ply_output_path, npz_output_path, sharpness_threshold):
    """
    Process a mesh file for sharp feature sampling.
    
    Pipeline:
      1. Import the mesh using Blender's OBJ importer.
      2. Switch to Edit mode and set the selection mode to EDGE.
      3. Select sharp edges using the given threshold.
      4. Switch to Object mode to access the selection.
      5. Create a bmesh from the imported mesh.
      6. Extract sharp edge data:
           - Record each sharp edge's vertex index pair.
           - Record the normals of adjacent faces and compute their dihedral angle.
      7. Collect unique sharp vertices and convert them to NumPy arrays.
      8. If the number of unique sharp vertices is less than half the target sample count,
         interpolate additional vertices along the sharp edges.
      9. Generate near-surface points around the sharp surface and compute their SDF.
     10. Sample coarse surface points and generate random space points; compute their SDF.
     11. Optionally perform farthest point sampling (FPS) on both surfaces.
     12. Save the sampled data to an NPZ file and, if sharp edges exist, save the sharp surface as a PLY file.
     13. Clean up by freeing resources and removing the imported object.
    
    Parameters:
      mesh_path: Input mesh file path.
      point_number: Target number of sample points.
      ply_output_path: Output PLY file path for sharp surface points.
      npz_output_path: Output NPZ file path for sample data.
      sharpness_threshold: Dihedral angle threshold (in radians) for selecting sharp edges.
    """
    # Step 1: Import the mesh.
    bpy.ops.wm.obj_import(filepath=mesh_path)
    obj = bpy.context.selected_objects[0]

    # Step 2: Switch to Edit mode.
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type="EDGE")

    # Step 3: Select sharp edges.
    bpy.ops.mesh.edges_select_sharp(sharpness=sharpness_threshold)

    # Step 4: Switch back to Object mode.
    bpy.ops.object.mode_set(mode='OBJECT')

    # Step 5: Create a bmesh for processing.
    bm = bmesh.new()
    bm.from_mesh(obj.data)

    # Step 6: Extract sharp edge data.
    sharp_edges = [edge for edge in bm.edges if edge.select]
    sharp_edges_vertices = []  # List of vertex index pairs.
    link_normal1 = []          # Normals from the first adjacent face.
    link_normal2 = []          # Normals from the second adjacent face.
    sharp_edges_angle = []     # Dihedral angles.
    vertices_set = set()       # Unique vertices on sharp edges.
    for edge in sharp_edges:
        vertices_set.update(edge.verts[:])
        sharp_edges_vertices.append([edge.verts[0].index, edge.verts[1].index])
        normal1 = edge.link_faces[0].normal
        normal2 = edge.link_faces[1].normal
        link_normal1.append(normal1)
        link_normal2.append(normal2)
        if normal1.length == 0.0 or normal2.length == 0.0:
            sharp_edges_angle.append(0.0)
        else:
            sharp_edges_angle.append(math.degrees(normal1.angle(normal2)))

    # Step 7: Collect unique sharp vertices.
    vertices = []
    vertices_index = []
    vertices_normal = []
    for vert in vertices_set:
        vertices.append(vert.co)
        vertices_index.append(vert.index)
        vertices_normal.append(vert.normal)
    vertices = np.array(vertices)
    vertices_index = np.array(vertices_index)
    vertices_normal = np.array(vertices_normal)

    sharp_edges_count = np.array(len(sharp_edges))
    sharp_edges_angle_array = np.array(sharp_edges_angle)
    if sharp_edges_count > 0:
        sharp_edge_link_normal = np.array(np.concatenate([link_normal1, link_normal2], axis=1))
        nan_mask = np.isnan(sharp_edge_link_normal)
        sharp_edge_link_normal = np.where(nan_mask, 0, sharp_edge_link_normal)
        nan_mask = np.isnan(vertices_normal)
        vertices_normal = np.where(nan_mask, 0, vertices_normal)
    sharp_edges_vertices_array = np.array(sharp_edges_vertices)

    # Step 8: Sample sharp surface points.
    if sharp_edges_count > 0:
        mesh_trimesh = trimesh.load(mesh_path, process=False)
        num_target_sharp_vertices = point_number // 2
        sharp_edge_length = sharp_edges_count
        sharp_edges_vertices_pair = sharp_edges_vertices_array
        sharp_vertices_pair = mesh_trimesh.vertices[sharp_edges_vertices_pair]  # Shape: (num_edges, 2, 3)
        epsilon = 1e-4
        edge_normal = 0.5 * sharp_edge_link_normal[:, :3] + 0.5 * sharp_edge_link_normal[:, 3:]
        norms = np.linalg.norm(edge_normal, axis=1, keepdims=True)
        norms = np.where(norms > epsilon, norms, epsilon)
        edge_normal = edge_normal / norms
        known_vertices = vertices
        known_vertices_normal = vertices_normal
        known_vertices = np.concatenate([known_vertices, known_vertices_normal], axis=1)
        num_known_vertices = known_vertices.shape[0]
        if num_known_vertices < num_target_sharp_vertices:
            num_new_vertices = num_target_sharp_vertices - num_known_vertices
            if num_new_vertices >= sharp_edge_length:
                num_new_vertices_per_pair = num_new_vertices // sharp_edge_length
                new_vertices = np.zeros((sharp_edge_length, num_new_vertices_per_pair, 6))
                start_vertex = sharp_vertices_pair[:, 0]
                end_vertex = sharp_vertices_pair[:, 1]
                for j in range(1, num_new_vertices_per_pair + 1):
                    t = j / float(num_new_vertices_per_pair + 1)
                    new_vertices[:, j - 1, :3] = (1 - t) * start_vertex + t * end_vertex
                    new_vertices[:, j - 1, 3:] = edge_normal
                new_vertices = new_vertices.reshape(-1, 6)
                remaining_vertices = num_new_vertices % sharp_edge_length
                if remaining_vertices > 0:
                    rng = np.random.default_rng()
                    ind = rng.choice(sharp_edge_length, remaining_vertices, replace=False)
                    new_vertices_remain = np.zeros((remaining_vertices, 6))
                    start_vertex = sharp_vertices_pair[ind, 0]
                    end_vertex = sharp_vertices_pair[ind, 1]
                    t = np.random.rand(remaining_vertices).reshape(-1, 1)
                    new_vertices_remain[:, :3] = (1 - t) * start_vertex + t * end_vertex
                    edge_normal_sel = 0.5 * sharp_edge_link_normal[ind, :3] + 0.5 * sharp_edge_link_normal[ind, 3:]
                    edge_normal_sel = edge_normal_sel / np.linalg.norm(edge_normal_sel, axis=1, keepdims=True)
                    new_vertices_remain[:, 3:] = edge_normal_sel
                    new_vertices = np.concatenate([new_vertices, new_vertices_remain], axis=0)
            else:
                remaining_vertices = num_target_sharp_vertices - num_known_vertices
                if remaining_vertices > 0:
                    rng = np.random.default_rng()
                    ind = rng.choice(sharp_edge_length, remaining_vertices, replace=False)
                    new_vertices_remain = np.zeros((remaining_vertices, 6))
                    start_vertex = sharp_vertices_pair[ind, 0]
                    end_vertex = sharp_vertices_pair[ind, 1]
                    t = np.random.rand(remaining_vertices).reshape(-1, 1)
                    new_vertices_remain[:, :3] = (1 - t) * start_vertex + t * end_vertex
                    edge_normal_sel = 0.5 * sharp_edge_link_normal[ind, :3] + 0.5 * sharp_edge_link_normal[ind, 3:]
                    edge_normal_sel = edge_normal_sel / np.linalg.norm(edge_normal_sel, axis=1, keepdims=True)
                    new_vertices_remain[:, 3:] = edge_normal_sel
                    new_vertices = new_vertices_remain
            target_vertices = np.concatenate([new_vertices, known_vertices], axis=0)
        else:
            target_vertices = known_vertices

        sharp_surface = target_vertices
        sharp_surface_points = sharp_surface[:, :3]
        
        # Step 9: Generate near-surface points around the sharp surface.
        sharp_near_surface_points = [
            sharp_surface_points + np.random.normal(scale=0.001, size=(len(sharp_surface_points), 3)),
            sharp_surface_points + np.random.normal(scale=0.005, size=(len(sharp_surface_points), 3)),
            sharp_surface_points + np.random.normal(scale=0.007, size=(len(sharp_surface_points), 3)),
            sharp_surface_points + np.random.normal(scale=0.01, size=(len(sharp_surface_points), 3))
        ]
        sharp_near_surface_points = np.concatenate(sharp_near_surface_points)
        f_sdf = SDF(mesh_trimesh.vertices, mesh_trimesh.faces)
        sharp_sdf = f_sdf(sharp_near_surface_points).reshape(-1, 1)
        sharp_near_surface = np.concatenate([sharp_near_surface_points, sharp_sdf], axis=1)
        
        # Step 10: Sample coarse surface points and generate random space points.
        coarse_surface_points, faces = mesh_trimesh.sample(200000, return_index=True)
        normals = mesh_trimesh.face_normals[faces]
        coarse_surface = np.concatenate([coarse_surface_points, normals], axis=1)
        coarse_near_surface_points = [
            coarse_surface_points + np.random.normal(scale=0.001, size=(len(coarse_surface_points), 3)),
            coarse_surface_points + np.random.normal(scale=0.005, size=(len(coarse_surface_points), 3))
        ]
        coarse_near_surface_points = np.concatenate(coarse_near_surface_points)
        space_points = np.random.uniform(-1.05, 1.05, (200000, 3))
        rand_points = np.concatenate([coarse_near_surface_points, space_points], axis=0)
        coarse_sdf = f_sdf(rand_points).reshape(-1, 1)
        rand_points = np.concatenate([rand_points, coarse_sdf], axis=1)
        
        # Optionally, re-sample the coarse surface.
        coarse_surface_points, faces = mesh_trimesh.sample(200000, return_index=True)
        normals = mesh_trimesh.face_normals[faces]
        coarse_surface = np.concatenate([coarse_surface_points, normals], axis=1)
        
        fps_coarse_surface_list = []
        for _ in range(1):
            kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(coarse_surface_points, num_target_sharp_vertices, h=5)
            fps_coarse_surface = coarse_surface[kdline_fps_samples_idx].reshape(-1, 1, 6)
            fps_coarse_surface_list.append(fps_coarse_surface)
        fps_coarse_surface = np.concatenate(fps_coarse_surface_list, axis=1)
        
        fps_sharp_surface_list = []
        if sharp_surface.shape[0] > num_target_sharp_vertices:
            for _ in range(1):
                kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(sharp_surface_points, num_target_sharp_vertices, h=5)
                fps_sharp_surface = sharp_surface[kdline_fps_samples_idx].reshape(-1, 1, 6)
                fps_sharp_surface_list.append(fps_sharp_surface)
            fps_sharp_surface = np.concatenate(fps_sharp_surface_list, axis=1)
        else:
            fps_sharp_surface = sharp_surface[:, None]

        sharp_surface[np.isinf(sharp_surface)] = 1
        sharp_surface[np.isnan(sharp_surface)] = 1
        fps_coarse_surface[np.isinf(fps_coarse_surface)] = 1
        fps_coarse_surface[np.isnan(fps_coarse_surface)] = 1
        
        np.savez(
            npz_output_path,
            fps_sharp_surface=fps_sharp_surface.astype(np.float32),
            sharp_near_surface=sharp_near_surface.astype(np.float32),
            fps_coarse_surface=fps_coarse_surface.astype(np.float32),
            rand_points=rand_points.astype(np.float32),
        )
    else:
        print(f"{npz_output_path} no sharp edges!")
    
    # Step 11: Save sharp surface points as a PLY file if sharp edges exist.
    if sharp_edges_count > 0:
        save_vertices_as_ply_open3d(sharp_surface[:, :3], ply_output_path)
    
    # Step 12: Cleanup.
    bm.free()
    del sharp_edges_angle_array, vertices, sharp_edges_count, sharp_edges_vertices_array, sharp_edges_vertices, sharp_edges_angle, sharp_edges
    bpy.data.objects.remove(obj, do_unlink=True)
    gc.collect()

def main_wrapper(input_folder, angle_threshold, point_number, sharp_point_path, sample_path) -> None:
    """
    Main entry point for processing meshes.

    This version of the sharp sample script continuously scans the provided input folder for unclaimed OBJ files.
    For each iteration, it:
      - Refreshes the list of OBJ files that do not start with "p_".
      - Randomly selects one file from that list.
      - Immediately renames the file (by prefixing with "p_") to mark it as claimed.
      - Processes the claimed file.
      - If processing fails, it renames the file back to its original name so it can be retried.
      - Continues until no unclaimed files remain.

    Parameters:
      input_folder: Directory containing input OBJ files.
      angle_threshold: Sharpness threshold angle (in degrees).
      point_number: Number of points to sample.
      sharp_point_path: Directory to save sharp point output (PLY files).
      sample_path: Directory to save sample data (NPZ files).
    """
    while True:
        # Refresh list of unclaimed OBJ files.
        files = [f for f in os.listdir(input_folder)
                 if f.lower().endswith('.obj') and not f.startswith("p_")]
        if not files:
            print("No unclaimed OBJ files found. Exiting.")
            break

        # Randomly select one file.
        selected_file = random.choice(files)
        input_path = os.path.join(input_folder, selected_file)
        claimed_file = "p_" + selected_file
        claimed_path = os.path.join(input_folder, claimed_file)
        try:
            os.rename(input_path, claimed_path)
            print(f"Claimed file: {selected_file} -> {claimed_file}")
        except Exception as e:
            print(f"Failed to claim {selected_file}: {e}")
            continue

        # Build output paths.
        ply_output_path = os.path.join(sharp_point_path, claimed_file.replace(".obj", ".ply"))
        npz_output_path = os.path.join(sample_path, claimed_file.replace(".obj", ".npz"))
        os.makedirs(sharp_point_path, exist_ok=True)
        os.makedirs(sample_path, exist_ok=True)

        try:
            process_mesh(claimed_path, point_number, ply_output_path, npz_output_path, math.radians(angle_threshold))
            gc.collect()
        except Exception as e:
            print(f"ERROR processing {ply_output_path}: {e}")
            # If processing fails, rename the file back (remove "p_" prefix) so it can be retried.
            try:
                os.rename(claimed_path, input_path)
                print(f"Reverted claim on file: {claimed_file} -> {selected_file}")
            except Exception as re:
                print(f"Failed to revert claim on {claimed_file}: {re}")
            gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="/data/input",
                        help="Path to the folder containing input OBJ files.")
    parser.add_argument("--angle_threshold", type=int, default=15,
                        help="Sharpness threshold angle (in degrees).")
    parser.add_argument("--point_number", type=int, default=65536,
                        help="Number of points to sample.")
    parser.add_argument("--sharp_point_path", type=str, default="/data/output/sharp_point_ply",
                        help="Directory to save the sharp point output (PLY files).")
    parser.add_argument("--sample_path", type=str, default="/data/output/sample",
                        help="Directory to save the sample data (NPZ files).")
    args, extras = parser.parse_known_args()
    main_wrapper(args.input_folder, args.angle_threshold, args.point_number, args.sharp_point_path, args.sample_path)
