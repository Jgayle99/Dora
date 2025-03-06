#!/usr/bin/env python3
import bpy
import os
import math
import argparse
import sys
import numpy as np
from mathutils import Vector, Matrix
from PIL import Image
import shutil

def log(message):
    print(message)
    sys.stdout.flush()

def parse_args():
    argv = sys.argv
    # Look for the separator so that we grab the correct arguments
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    parser = argparse.ArgumentParser(
        description="Render multi-view depth and normal maps using bpy (Eevee or Cycles)"
    )
    parser.add_argument("--input_folder", type=str, default="D://trainingModels/output_subsets/output_mix_5",
                        help="Path to input normalized OBJ meshes")
    parser.add_argument("--output_folder", type=str, default="D://processed",
                        help="Folder to save rendered outputs")
    parser.add_argument("--resolution", type=int, default=1024,
                        help="Render resolution (square)")
    parser.add_argument("--num_views", type=int, default=36,
                        help="Number of views to render")
    parser.add_argument("--cam_dist", type=float, default=2.0,
                        help="Camera distance from object")
    parser.add_argument("--mesh_file", type=str, default=None,
                        help="If provided, process only this OBJ file")
    return parser.parse_args(argv)

def clear_scene():
    log("Clearing scene...")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # Ensure no lights are present
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)


def enable_gpus(device_type, use_cpus=False):
    log(f"Enabling GPUs with device type: {device_type}, use_cpus: {use_cpus}") # Added log
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cycles_preferences.refresh_devices()
    devices = cycles_preferences.devices

    if not devices:
        raise RuntimeError("No render devices found!") # Improved error message

    activated_gpus = []
    for device in devices:
        if device.type == "CPU":
            device.use = use_cpus
            if use_cpus:
                log(f"  Using CPU device: {device.name}") # Log CPU usage if enabled
            else:
                log(f"  Disabling CPU device: {device.name}")
        else: # For GPU devices
            device.use = True
            activated_gpus.append(device.name)
            log(f"  Activated GPU device: {device.name}")

    cycles_preferences.compute_device_type = device_type
    bpy.context.scene.cycles.device = "GPU"
    log(f"Set Cycles device type to: {device_type}, scene device to GPU.") # Added log for device type setting

    return activated_gpus


def setup_scene(resolution, engine='CYCLES'):
    log(f"Setting up scene (resolution: {resolution}, engine: {engine})")
    scene = bpy.context.scene
    scene.render.engine = engine
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.view_layers["ViewLayer"].use_pass_z = True
    scene.view_layers["ViewLayer"].use_pass_normal = True
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_depth = '32'

    if engine == 'CYCLES':
        log("Configuring Cycles GPU rendering...")
        try:
            activated_gpus = enable_gpus("CUDA") # Call enable_gpus function
            log(f"Activated GPUs: {activated_gpus}") # Log activated GPUs
        except RuntimeError as e:
            log(f"GPU setup error: {e}. Falling back to CPU.") # Log error and fallback message
            scene.cycles.device = 'CPU' # Fallback to CPU if GPU setup fails
            scene.cycles.device = 'CPU'
        scene.cycles.samples = 256
        # --- Redundant/Duplicated code below - consider removing if enable_gpus is sufficient ---
        # (However, keeping it for now in case specific RTX selection is still desired AFTER general GPU enabling)
        bpy.context.scene.cycles.device = 'GPU' # Redundant after enable_gpus, but harmless
        # bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA' # Already set in enable_gpus
        for device in bpy.context.preferences.addons['cycles'].preferences.devices:
            if "RTX" in device.name:
                device.use = True
            else:
                device.use = False
        # --- End of potentially redundant code ---

    # Disable world lighting to remove shadows from normal maps - Robust version.
    if bpy.data.worlds:
        bpy.data.worlds.remove(bpy.data.worlds["World"], do_unlink=True)
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    bg_tree = world.node_tree
    bg_tree.nodes.clear()

    bg_node = bg_tree.nodes.new(type='ShaderNodeBackground')
    bg_node.inputs['Color'].default_value = (0, 0, 0, 1)
    bg_node.inputs['Strength'].default_value = 0.0

    world_output_node = bg_tree.nodes.new(type='ShaderNodeOutputWorld')
    bg_tree.links.new(bg_node.outputs['Background'], world_output_node.inputs['Surface'])

    scene.world = world
    return scene



# def setup_scene(resolution, engine='CYCLES'):
#     log(f"Setting up scene (resolution: {resolution}, engine: {engine})")
#     scene = bpy.context.scene
#     scene.render.engine = engine
#     scene.render.resolution_x = resolution
#     scene.render.resolution_y = resolution
#     scene.render.resolution_percentage = 100
#     scene.render.film_transparent = True
#     scene.view_layers["ViewLayer"].use_pass_z = True
#     scene.view_layers["ViewLayer"].use_pass_normal = True
#     scene.render.image_settings.file_format = 'OPEN_EXR'
#     scene.render.image_settings.color_depth = '32'

#     # --- OPTIONAL: Use Cycles with GPU instead of EEVEE ---
#     # Uncomment the following block to switch to Cycles and force GPU rendering.
#     # Note that switching engines may change render appearance and speed.
#     scene.cycles.samples = 64  # Use the explicit 'scene' object here
#     bpy.context.scene.cycles.device = 'GPU'
#     # Set compute device type; choose 'CUDA' (or 'OPTIX' if supported)
#     bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
#     # Enable only the high-performance GPU (adjust the device name as needed)
#     for device in bpy.context.preferences.addons['cycles'].preferences.devices:
#         if "RTX" in device.name:  # or match your GPU name exactly, e.g. "GeForce RTX 3090"
#             device.use = True
#         else:
#             device.use = False
    
#     # Disable world lighting to remove shadows from normal maps.
#     if bpy.data.worlds:
#         world = bpy.data.worlds[0]
#         world.use_nodes = True
#         bg = world.node_tree.nodes.get("Background")
#         if bg:
#             bg.inputs[0].default_value = (0, 0, 0, 1)  # Black color
#             bg.inputs[1].default_value = 0.0           # Zero strength
#     return scene

# def setup_scene(resolution, engine='BLENDER_EEVEE'):
#     log(f"Setting up scene (resolution: {resolution}, engine: {engine})")
#     scene = bpy.context.scene
#     scene.render.engine = engine
#     scene.render.resolution_x = resolution
#     scene.render.resolution_y = resolution
#     scene.render.resolution_percentage = 100
#     scene.render.film_transparent = True
#     scene.view_layers["ViewLayer"].use_pass_z = True
#     scene.view_layers["ViewLayer"].use_pass_normal = True
#     scene.render.image_settings.file_format = 'OPEN_EXR'
#     scene.render.image_settings.color_depth = '32'
    
#     # Set Eevee specific settings:
#     scene.eevee.taa_render_samples = 64   # Use 256 TAA samples
#     scene.eevee.use_gtao = False            # Disable ambient occlusion
#     scene.eevee.use_bloom = False           # Disable bloom

#     # Disable world lighting to remove shadows.
#     if bpy.data.worlds:
#         world = bpy.data.worlds[0]
#         world.use_nodes = True
#         bg = world.node_tree.nodes.get("Background")
#         if bg:
#             bg.inputs[0].default_value = (0, 0, 0, 1)  # Black color
#             bg.inputs[1].default_value = 0.0           # Zero strength
#     return scene

def import_mesh(filepath):
    log(f"Importing mesh from: {filepath}")
    bpy.ops.wm.obj_import(filepath=filepath)
    obj = bpy.context.selected_objects[0]
    log(f"Mesh imported: {obj.name}")
    return obj

def set_normal_material(obj):
    log(f"Setting normal material for object: {obj.name}")
    mat = bpy.data.materials.new(name="NormalOnly")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    geom_node = nodes.new("ShaderNodeNewGeometry")
    geom_node.location = (-200, 0)
    mapping_node = nodes.new("ShaderNodeMapping")
    mapping_node.location = (-50, 0)
    mapping_node.inputs["Scale"].default_value = (0.5, 0.5, 0.5)
    mapping_node.inputs["Location"].default_value = (0.5, 0.5, 0.5)
    links.new(geom_node.outputs["Normal"], mapping_node.inputs["Vector"])
    emission_node = nodes.new("ShaderNodeEmission")
    emission_node.location = (100, 0)
    links.new(mapping_node.outputs["Vector"], emission_node.inputs["Color"])
    output_node = nodes.new("ShaderNodeOutputMaterial")
    output_node.location = (300, 0)
    links.new(emission_node.outputs["Emission"], output_node.inputs["Surface"])
    return obj

def setup_camera():
    log("Setting up camera...")
    cam = None
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            cam = obj
            break
    if cam is None:
        bpy.ops.object.camera_add()
        cam = bpy.context.active_object
        log(f"Camera created: {cam.name}")
    bpy.context.scene.camera = cam
    cam.data.lens = 35
    cam.data.clip_start = 0.01
    cam.data.clip_end = 100
    return cam

# def compute_camera_poses(num_views, cam_dist, elevation_deg=30):
#     log(f"Computing {num_views} camera poses (cam_dist={cam_dist}, elevation={elevation_deg}°)")
#     poses = []
#     elev_rad = math.radians(elevation_deg)
#     for i in range(num_views):
#         azimuth_rad = math.radians(360.0 * i / num_views)
#         x = cam_dist * math.cos(azimuth_rad) * math.cos(elev_rad)
#         y = cam_dist * math.sin(elev_rad)
#         z = cam_dist * math.sin(azimuth_rad) * math.cos(elev_rad)
#         pos = Vector((x, y, z))
#         direction = -pos
#         rot_quat = direction.to_track_quat('-Z', 'Y')
#         cam_mat = Matrix.Translation(pos) @ rot_quat.to_matrix().to_4x4()
#         poses.append(cam_mat)
#     return poses

# Updated compute_camera_poses using a Fibonacci sphere distribution
def compute_camera_poses(num_views, cam_dist):
    log(f"Computing {num_views} camera poses (cam_dist={cam_dist}) using Fibonacci sphere distribution.")
    poses = []
    offset = 2.0 / num_views
    increment = math.pi * (3.0 - math.sqrt(5.0))  # golden angle in radians
    for i in range(num_views):
        # y goes from -1 to 1
        y = ((i * offset) - 1) + (offset / 2)
        r = math.sqrt(max(0.0, 1 - y * y))
        phi = i * increment
        x = math.cos(phi) * r
        z = math.sin(phi) * r
        pos = Vector((x, y, z)) * cam_dist
        direction = -pos
        rot_quat = direction.to_track_quat('-Z', 'Y')
        cam_mat = Matrix.Translation(pos) @ rot_quat.to_matrix().to_4x4()
        poses.append(cam_mat)
    return poses

def setup_compositor_nodes(output_folder_mesh, view_idx, mesh_name):
    log(f"Setting up compositor nodes for view {view_idx:03d} in folder: {output_folder_mesh}")
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    for node in list(tree.nodes):
        tree.nodes.remove(node)
    rl_node = tree.nodes.new("CompositorNodeRLayers")
    rl_node.location = (0, 0)
    map_node = tree.nodes.new("CompositorNodeMapValue")
    map_node.location = (200, -100)
    map_node.use_min = True
    map_node.min = [0]
    map_node.use_max = True
    map_node.max = [10]
    map_node.size = [0.05]
    map_node.offset = [0]
    depth_node = tree.nodes.new("CompositorNodeOutputFile")
    depth_node.label = "Depth Output"
    depth_node.base_path = output_folder_mesh
    depth_node.format.file_format = "OPEN_EXR"
    depth_node.format.color_depth = "32"
    depth_node.file_slots[0].use_node_format = True
    depth_node.file_slots[0].path = f"{mesh_name}_view_{view_idx:03d}_depth"
    depth_node.file_slots[0].save_as_render = False
    normal_node = tree.nodes.new("CompositorNodeOutputFile")
    normal_node.label = "Normal Output"
    normal_node.base_path = output_folder_mesh
    normal_node.format.file_format = "OPEN_EXR"
    normal_node.format.color_depth = "16"
    normal_node.file_slots[0].use_node_format = True
    normal_node.file_slots[0].path = f"{mesh_name}_view_{view_idx:03d}_normal"
    normal_node.file_slots[0].save_as_render = False
    links = tree.links
    links.new(rl_node.outputs["Depth"], map_node.inputs[0])
    links.new(map_node.outputs["Value"], depth_node.inputs[0])
    links.new(rl_node.outputs["Normal"], normal_node.inputs[0])

def remove_compositor_nodes():
    log("Removing compositor nodes...")
    scene = bpy.context.scene
    if scene.use_nodes:
        tree = scene.node_tree
        for node in list(tree.nodes):
            tree.nodes.remove(node)

def render_views(mesh_path, resolution, num_views, output_folder, cam_dist):
    try:
        log(f"=== Starting render for mesh: {mesh_path} ===")
        clear_scene()
        scene = setup_scene(resolution, engine='BLENDER_EEVEE')
        cam = setup_camera()
        obj = import_mesh(mesh_path)
        mesh_name_with_ext = os.path.basename(mesh_path)
        mesh_name = os.path.splitext(mesh_name_with_ext)[0]
        bpy.context.view_layer.update()
        poses = compute_camera_poses(num_views, cam_dist)
        output_folder_mesh = os.path.join(output_folder, mesh_name)
        log(f"Creating output folder: {output_folder_mesh}")
        os.makedirs(output_folder_mesh, exist_ok=True)
        original_materials = obj.data.materials[:]
        for i, pose in enumerate(poses):
            log(f"Rendering view {i:03d} for mesh '{mesh_name}'")
            cam.matrix_world = pose
            bpy.context.view_layer.update()
            set_normal_material(obj)
            bpy.context.view_layer.update()
            setup_compositor_nodes(output_folder_mesh, i, mesh_name)
            bpy.ops.render.render(write_still=True)
            remove_compositor_nodes()
            obj.data.materials.clear()
            for mat in original_materials:
                obj.data.materials.append(mat)
            bpy.context.view_layer.update()
            log(f"Completed view {i:03d}")
        
        # Write the marker OBJ by copying the original file.
        mesh_marker_path = os.path.join(output_folder_mesh, f"{mesh_name}.obj")
        log(f"DEBUG: Writing marker (copying OBJ) => {mesh_marker_path}")
        try:
            shutil.copy2(mesh_path, mesh_marker_path)
            log("Copied original OBJ as marker successfully.")
        except Exception as e:
            log(f"Error copying OBJ file: {e}")
        
        # Clean up: remove object and mesh data.
        obj_mesh_name = obj.data.name
        bpy.data.objects.remove(obj, do_unlink=True)
        if obj_mesh_name in bpy.data.meshes:
            bpy.data.meshes.remove(bpy.data.meshes[obj_mesh_name], do_unlink=True)
        log(f"=== Finished render for mesh: {mesh_path} ===")
    except Exception as ex:
        log(f"Exception during rendering: {ex}")

def main():
    try:
        log("Render script started.")
        args = parse_args()
        log(f"Arguments: {args}")
        os.makedirs(args.output_folder, exist_ok=True)

        if args.mesh_file:
            mesh_files = [args.mesh_file]
            log(f"Processing single mesh file: {args.mesh_file}")
        else:
            mesh_files = sorted([
                os.path.join(args.input_folder, f)
                for f in os.listdir(args.input_folder)
                if f.lower().endswith(".obj")
            ])
            log(f"Found {len(mesh_files)} OBJ files in input folder.")

        if not mesh_files:
            log("No OBJ files found in input folder.")
            return

        for mesh_path in mesh_files:
            render_views(mesh_path, resolution=args.resolution, num_views=args.num_views,
                         output_folder=args.output_folder, cam_dist=args.cam_dist)

        log("All rendering complete.")
    except Exception as e:
        log(f"Exception in main: {e}")

if __name__ == "__main__":
    main()
    # Optionally, uncomment the next line to quit Blender automatically:
    # bpy.ops.wm.quit_blender()
