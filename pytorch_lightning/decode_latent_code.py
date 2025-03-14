import torch
import numpy as np
import os
import argparse
from model_loader import load_michelangelo_vae_model


def decode_latent(output_folder, pretrained_model_path, config_path, latent_npz_path, output_file_prefix="decoded", export_mesh=False, export_xyz=False, octree_depth=5):
    """
    Decodes latent code from an NPZ file into a point cloud (XYZ format) / mesh (OBJ) using extract_geometry_by_diffdmc.
    """

    print(f"\n--- Decoding Latent from NPZ: {latent_npz_path} ---")

    # 1. Load VAE Model
    autoencoder = load_michelangelo_vae_model(pretrained_model_path, config_path)
    # Ensure that the model is on the correct device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = autoencoder.to(device)
    print("\n--- 1. VAE Model Loaded and moved to device ---")

    # 2. Load Latent Code from NPZ
    try:
        latent_data = np.load(latent_npz_path)

        # Assuming the latent code is stored under key 'latent_code' in the NPZ file
        if "latent_code" in latent_data:
            latent_code_np = latent_data["latent_code"]
        else:
            raise KeyError("Could not find a key 'latent_code' in the NPZ file. Please check the file.")

        latent_code_tensor = torch.from_numpy(latent_code_np).float()
        print("\n--- 2. Latent Code Loaded from NPZ ---")
        print(f"Shape of loaded latent_code (numpy): {latent_code_np.shape}")
        print(f"Shape of loaded latent_code_tensor: {latent_code_tensor.shape}")
        
        # Move the latent code tensor to GPU if available
        latent_code_tensor = latent_code_tensor.to(device)
        print(f"Moved latent_code_tensor to device: {device}")

        # 3. Extract Geometry using extract_geometry_by_diffdmc 
        with torch.no_grad():
            print(f"Shape of latent_code_tensor before extract_geometry_by_diffdmc: {latent_code_tensor.shape}")
            
            # Ensure the latent_code_tensor is passed to the device
            mesh_v_f, has_surface = autoencoder.extract_geometry_by_diffdmc(latent_code_tensor, octree_depth=octree_depth)

        print("\n--- 4. Geometry Extracted using extract_geometry_by_diffdmc ---")
        print(f"mesh_v_f: {mesh_v_f}")
        print(f"has_surface: {has_surface}")

        # 5. Save Decoded Point Cloud (vertices of the mesh) as XYZ and OBJ
        if mesh_v_f and has_surface[0]:
            vertices, faces = mesh_v_f[0]
            print("\n--- 5. Saving Decoded Point Cloud (xyz format) and Mesh (obj) ---")
            print(f"Shape of vertices (numpy): {vertices.shape}")
            
            # Move vertices to CPU and then convert to numpy
            vertices_cpu = vertices.cpu().numpy()
            
            # Save Point Cloud (XYZ file)
            if export_xyz:
                decoded_pc_path = os.path.join(output_folder, f"{output_file_prefix}_decoded_point_cloud.xyz")
                np.savetxt(decoded_pc_path, vertices_cpu, fmt="%.6f")  # Save mesh vertices as XYZ
                print(f"Decoded point cloud (mesh vertices) saved to: {decoded_pc_path}")
            
            # Save Mesh as OBJ file
            if export_mesh:
                obj_file_path = os.path.join(output_folder, f"{output_file_prefix}_mesh.obj")
                with open(obj_file_path, "w") as obj_file:
                    # Write vertices
                    for vertex in vertices_cpu:
                        obj_file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
                    
                    # Write faces (indices are 1-based in OBJ, so add 1)
                    for face in faces.cpu().numpy():  # Move faces to CPU if on GPU
                        obj_file.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

                print(f"Decoded mesh (OBJ file) saved to: {obj_file_path}")
        else:
            print("\n--- 5. No Surface Extracted or Invalid Mesh, cannot save point cloud or mesh ---")

    except FileNotFoundError:
        print(f"Error: NPZ file not found at {latent_npz_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Decode latent code into point cloud and mesh.")
    parser.add_argument('--export_mesh', type=bool, default=True, help='Export the mesh as an OBJ file')
    parser.add_argument('--export_xyz', type=bool, default=True, help='Export the point cloud as an XYZ file')
    parser.add_argument('--octree_depth', type=int, default=8, help='Octree depth for geometry extraction')
    parser.add_argument('--latent_npz_path', type=str, required=True, help="Path to the latent NPZ file")
    parser.add_argument('--output_folder', type=str, required=True, help="Folder to save the results")
    parser.add_argument('--pretrained_model', type=str, required=True, help="Path to the pretrained model")
    parser.add_argument('--config_path', type=str, required=True, help="Path to the config file")
    
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    decode_latent(
        output_folder=args.output_folder,
        pretrained_model_path=args.pretrained_model,
        config_path=args.config_path,
        latent_npz_path=args.latent_npz_path,
        output_file_prefix="decoded",
        export_mesh=args.export_mesh,
        export_xyz=args.export_xyz,
        octree_depth=args.octree_depth
    )

    print("\nDecoding process finished.")


if __name__ == "__main__":
    main()
