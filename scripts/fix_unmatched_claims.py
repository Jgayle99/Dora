import os
import argparse

def remove_prefix_from_unmatched(input_folder, output_folder):
    """
    For each OBJ file in the input folder that starts with "p_",
    check if a corresponding PLY file exists in the output folder.
    If not, remove the "p_" prefix from the OBJ file (i.e. rename it),
    so that it can be reprocessed later.
    
    Parameters:
      input_folder (str): Folder containing claimed OBJ files (prefixed with "p_").
      output_folder (str): Folder containing the output PLY files.
    """
    # List all OBJ files in the input folder that are claimed (i.e. start with "p_")
    claimed_obj_files = [f for f in os.listdir(input_folder) 
                         if f.lower().endswith('.obj') and f.startswith("p_")]
    
    total_claimed = len(claimed_obj_files)
    print(f"Found {total_claimed} claimed OBJ files in {input_folder}.")
    renamed_count = 0
    
    for obj_file in claimed_obj_files:
        # The corresponding PLY file should have the same base name (including the "p_" prefix)
        base_name = os.path.splitext(obj_file)[0]  # e.g. "p_256_sf-123"
        corresponding_ply = base_name + ".ply"       # e.g. "p_256_sf-123.ply"
        output_path = os.path.join(output_folder, corresponding_ply)
        
        # If no matching PLY file exists, remove the "p_" prefix
        if not os.path.exists(output_path):
            original_name = obj_file[2:]  # Remove first two characters "p_"
            old_path = os.path.join(input_folder, obj_file)
            new_path = os.path.join(input_folder, original_name)
            try:
                os.rename(old_path, new_path)
                print(f"Renamed '{obj_file}' to '{original_name}' (no matching output found).")
                renamed_count += 1
            except Exception as e:
                print(f"Error renaming '{obj_file}': {e}")
    
    print(f"Completed. Renamed {renamed_count} files out of {total_claimed} claimed files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="For any OBJ file in the input folder (prefixed with 'p_') that doesn't have a matching PLY file in the output folder, remove the 'p_' prefix."
    )
    parser.add_argument("--input_folder", type=str, default="D://trainingModels/output_subsets/output_mix_5",
                        help="Path to the folder containing input OBJ files (with 'p_' prefix).")
    parser.add_argument("--output_folder", type=str, default="D://training/output/sharp_point_ply",
                        help="Path to the folder containing output PLY files.")
    args = parser.parse_args()
    remove_prefix_from_unmatched(args.input_folder, args.output_folder)
