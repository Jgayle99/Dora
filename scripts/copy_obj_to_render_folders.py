#!/usr/bin/env python3
import os
import shutil
import argparse

def copy_obj_to_render_folders(input_folder, output_folder):
    """
    Copies OBJ files from input_folder to their corresponding render folders
    in output_folder, based on the folder naming convention.
    """
    mesh_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".obj")]
    if not mesh_files:
        print(f"No OBJ files found in input folder: {input_folder}")
        return

    print(f"Copying OBJ files to render folders in: {output_folder}")

    for mesh_file in mesh_files:
        mesh_name_no_ext = os.path.splitext(mesh_file)[0] # Get filename without extension, KEEPING prefixes like "256_" or "p_256_"
        render_folder_name = mesh_name_no_ext # Render folder name is the same as the mesh name (without extension)
        render_folder_path = os.path.join(output_folder, render_folder_name)

        input_mesh_path = os.path.join(input_folder, mesh_file)
        output_mesh_path = os.path.join(render_folder_path, f"{render_folder_name}.obj") # OBJ filename inside render folder

        if not os.path.exists(render_folder_path):
            print(f"Warning: Render folder not found for {mesh_file}: {render_folder_path}. Skipping.")
            continue

        if os.path.exists(output_mesh_path):
            print(f"OBJ file already exists in render folder for {mesh_file}: {output_mesh_path}. Skipping copy.")
            continue

        try:
            shutil.copy2(input_mesh_path, output_mesh_path) # copy2 preserves metadata
            print(f"Copied OBJ for {mesh_file} to: {output_mesh_path}")
        except Exception as e:
            print(f"Error copying OBJ for {mesh_file} to {output_mesh_path}: {e}")

    print("OBJ copying process complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Copy OBJ files to their corresponding render folders."
    )
    parser.add_argument("--input_folder", type=str, default="D://trainingModels/output_subsets/output_mix_5",
                        help="Path to the folder containing input OBJ files.")
    parser.add_argument("--output_folder", type=str, default="D://processed",
                        help="Path to the base output folder where render folders are located.")
    args = parser.parse_args()

    copy_obj_to_render_folders(args.input_folder, args.output_folder)

if __name__ == "__main__":
    main()