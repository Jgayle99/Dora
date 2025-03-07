#!/usr/bin/env python
"""
This script loads a 3D mesh from an OBJ file, renders multiple normal and depth maps
from different camera viewpoints using OpenGL, and saves the resulting images.
It uses argparse to allow users to pass rendering parameters from the command line.
Rendered files are prefixed with the base name of the OBJ file.
"""
# import ctypes
# # Make sure to use the correct path to your DLL.
# ctypes.CDLL("D:\\dev\\blender_shapenet_render\\gpu_select.dll")

import argparse  # For command-line argument parsing
import numpy as np
import OpenGL.GL as gl
import OpenGL.GLUT as glut
import OpenGL.GLU as glu
import cv2
import trimesh
from PIL import Image
import time
from view_utils import compute_camera_poses  # Custom function to compute camera poses
import os

# Remove the DISPLAY environment variable if present to prevent freeglut errors.
# This step is necessary to avoid issues related to display parameters on some systems.
try:
    del os.environ['DISPLAY']
except KeyError:
    pass

def create_maps(obj_file, output_folder, num_views, cam_dist, multisample, resolution):
    """
    Loads a mesh from an OBJ file, renders multiple normal and depth maps,
    and saves the results as PNG files.

    Args:
        obj_file (str): Path to the OBJ file.
        output_folder (str): Folder to save the rendered images.
        num_views (int): Number of different views to render.
        cam_dist (float): Distance of the camera from the object.
        multisample (bool): Enable/Disable multi-sampling for smoother results.
        resolution (int): Resolution (width and height) of the maps.
    """
    try:
        # Load the mesh and extract vertices, faces, and vertex normals.
        mesh = trimesh.load(obj_file)
        vertices = mesh.vertices.astype(np.float32)
        faces = mesh.faces.astype(np.uint32)
        normals = mesh.vertex_normals.astype(np.float32)
    except Exception as e:
        print(f"Error loading OBJ file: {e}")
        return

    # Extract the base name of the OBJ file (without directory and extension)
    obj_base = os.path.splitext(os.path.basename(obj_file))[0]

    # Set the width and height of the rendered images based on the resolution parameter.
    width, height = resolution, resolution

    # Record the start time to later measure total execution time.
    start_time = time.time()

    # Initialize GLUT for window management and rendering context.
    glut.glutInit()

    # Set display mode. Enable multisampling if requested to smooth out edges.
    if multisample:
        glut.glutInitDisplayMode(glut.GLUT_RGBA | glut.GLUT_DOUBLE | glut.GLUT_DEPTH | glut.GLUT_MULTISAMPLE)
    else:
        glut.glutInitDisplayMode(glut.GLUT_RGBA | glut.GLUT_DOUBLE | glut.GLUT_DEPTH)
    
    # Position the window off-screen (e.g., top-left corner far outside the visible area).
    glut.glutInitWindowPosition(-10000, -10000)
    # Set the window size and create the window.
    glut.glutInitWindowSize(width, height)
    glut.glutCreateWindow(b"Normal and Depth Map Renderer")
    glut.glutHideWindow()  # This hides the window from the screen.
    
    # If multisampling is enabled, activate the multisample flag and set a hint for higher quality.
    if multisample:
        gl.glEnable(gl.GL_MULTISAMPLE)
        gl.glHint(gl.GL_POLYGON_SMOOTH_HINT, gl.GL_NICEST)

    # Enable depth testing for proper occlusion.
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glDepthFunc(gl.GL_LEQUAL)

    # Define the viewport to match the window size.
    gl.glViewport(0, 0, width, height)

    # Set the clear color to white.
    gl.glClearColor(1.0, 1.0, 1.0, 1.0)

    # Set up the projection matrix with a perspective projection.
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    glu.gluPerspective(60.0, width / height, 0.05, 15.0)
    gl.glMatrixMode(gl.GL_MODELVIEW)

    # Compute camera poses for the specified number of views and camera distance.
    poses = compute_camera_poses(num_views, cam_dist)

    # Swap buffers to ensure the context is ready for rendering.
    glut.glutSwapBuffers()

    # Loop through each computed camera pose to render the mesh from different viewpoints.
    # Extend the poses list by one extra iteration using the last pose.
    for idx, cam_mat in enumerate(poses + [poses[-1]]):
        # Extract the camera position (cx, cy, cz) from the camera matrix.
        cx, cy, cz = cam_mat[:3, 3]
        gl.glLoadIdentity()
        # Position the camera using gluLookAt; the camera always looks at the origin.
        glu.gluLookAt(cx, cy, cz, 0, 0, 0, 0, 1, 0)
        # Clear the current frame buffers.
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # Enable and set the vertex pointer for the mesh vertices.
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glVertexPointer(3, gl.GL_FLOAT, 0, vertices)

        # Enable and set the color pointer.
        # The vertex normals are normalized to [0, 1] and used as colors.
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)
        normal_colors = (normals + 1.0) / 2.0
        gl.glColorPointer(3, gl.GL_FLOAT, 0, normal_colors)

        # Draw the mesh using triangles.
        gl.glDrawElements(gl.GL_TRIANGLES, len(faces.flatten()), gl.GL_UNSIGNED_INT, faces)
        
        # Swap buffers to update the window with the rendered image.
        glut.glutSwapBuffers()

        # Save the normal map image (skip the first iteration due to buffer swap timing).
        if idx > 0:
            pixels = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
            img_data = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, 3))
            # OpenGL's coordinate system has the origin at the bottom left so flip vertically.
            img_data = cv2.flip(img_data, 0)
            Image.fromarray(img_data, mode="RGB").save(f"{output_folder}\\{obj_base}_view_{idx-1:03d}_normal.png")
        
        # Save the depth map image for the current view.
        if idx < num_views:
            depth_pixels = gl.glReadPixels(0, 0, width, height, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT)
            depth_data = np.frombuffer(depth_pixels, dtype=np.float32).reshape((height, width))
            depth_data = cv2.flip(depth_data, 0)
            # Normalize depth values to the range [0, 255] for visualization.
            depth_data = ((depth_data - depth_data.min()) / (depth_data.max() - depth_data.min()) * 255).astype(np.uint8)
            Image.fromarray(depth_data, mode="L").save(f"{output_folder}\\{obj_base}_view_{idx:03d}_depth.png")

    # Exit the GLUT main loop after rendering is complete.
    glut.glutLeaveMainLoop()
    
    # Print the total execution time.
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

def str2bool(v):
    """
    Convert a string representation of truth to a boolean value.
    This is useful for parsing boolean command-line arguments.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    # Create the argument parser to handle command-line inputs.
    parser = argparse.ArgumentParser(
        description="Render normal and depth maps from a 3D OBJ model using OpenGL."
    )
    
    # Define the expected command-line arguments with defaults.
    parser.add_argument(
        '--obj_file', type=str, default="mesh.obj",
        help="Path to the OBJ file to be rendered."
    )
    # Output folder argument for saving renders.
    parser.add_argument(
        '--output_folder', type=str, default="output",
        help="Path to the render output folder."
    )
    parser.add_argument(
        '--num_views', type=int, default=36,
        help="Number of different views (angles) to render."
    )
    parser.add_argument(
        '--cam_dist', type=float, default=2.0,
        help="Distance of the camera from the object."
    )
    parser.add_argument(
        '--multisample', type=str2bool, nargs='?', const=True, default=True,
        help="Enable multi-sampling for smoother results (True/False)."
    )
    parser.add_argument(
        '--resolution', type=int, default=256,
        help="Resolution (width and height) of the generated maps."
    )
    
    # Parse the command-line arguments.
    args = parser.parse_args()
    
    # Create the output folder if it doesn't exist.
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder, exist_ok=True)
    
    # Call the create_maps function with the arguments provided by the user.
    create_maps(args.obj_file, args.output_folder, args.num_views, args.cam_dist, args.multisample, args.resolution)
