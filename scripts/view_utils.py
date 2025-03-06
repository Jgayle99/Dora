import numpy as np
import math

def compute_camera_poses(num_views, cam_dist):
    """
    Computes camera positions using a Fibonacci sphere distribution to ensure 
    uniform coverage around the target object, including views from underneath.
    
    The function creates 'num_views' camera poses placed uniformly on a sphere
    of radius 'cam_dist' around the origin (assumed center of the mesh).
    
    Args:
        num_views (int): The number of camera views to generate.
        cam_dist (float): The distance from the target object to place the cameras.
        
    Returns:
        List of 4x4 numpy arrays representing the transformation matrices (poses)
        for each camera view.
    """
    poses = []  # List to store each computed camera pose as a 4x4 matrix.
    
    # 'offset' determines the incremental step in the y-axis for each view.
    offset = 2.0 / num_views
    
    # 'increment' is set to the golden angle in radians, which helps distribute points uniformly on the sphere.
    increment = math.pi * (3.0 - math.sqrt(5.0))
    
    for i in range(num_views):
        # Compute the y coordinate to evenly space views from -1 to 1.
        y = ((i * offset) - 1) + (offset / 2)
        # Calculate the radius of the circle at height y on the sphere.
        r = math.sqrt(max(0.0, 1 - y * y))
        # Calculate the azimuthal angle for this view.
        phi = i * increment
        # Compute the x and z coordinates based on the azimuthal angle and circle radius.
        x = math.cos(phi) * r
        z = math.sin(phi) * r
        # Scale the (x, y, z) position by the camera distance.
        pos = np.array([x, y, z]) * cam_dist
        
        # Determine the direction vector from the camera position to the origin (target).
        direction = -pos
        direction = direction / np.linalg.norm(direction)  # Normalize to unit length
        
        # Compute the rotation quaternion that will orient the camera to look at the origin.
        rot_quat = compute_rotation_quaternion(direction)
        
        # Build the full camera transformation matrix: a translation followed by the rotation.
        cam_mat = translation_matrix(pos) @ rotation_matrix_from_quaternion(rot_quat)
        poses.append(cam_mat)
        
    return poses

def compute_rotation_quaternion(direction):
    """
    Computes a quaternion representing the rotation required to align the camera's 
    default viewing direction (assumed to be -Z) with the given 'direction'.
    
    This is done by constructing an orthonormal basis using the given direction,
    a global up vector, and the right vector computed via the cross product.
    
    Args:
        direction (np.array): The desired view direction (unit vector) from the camera to the target.
    
    Returns:
        A numpy array representing the quaternion (w, x, y, z) for the rotation.
    """
    # Define a global up vector.
    up = np.array([0, 1, 0])
    
    # We want the camera's -Z axis to align with the desired view direction.
    z_axis = -direction
    
    # Compute the right vector (x_axis) as the cross product between 'up' and 'z_axis'.
    x_axis = np.cross(up, z_axis)
    # If 'up' and 'z_axis' are nearly parallel, use a default right vector.
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.array([1, 0, 0])
    x_axis /= np.linalg.norm(x_axis)  # Normalize the right vector
    
    # Compute the true up vector (y_axis) as the cross product of 'z_axis' and 'x_axis'.
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)  # Normalize the up vector
    
    # Construct a rotation matrix using the orthonormal basis (x_axis, y_axis, z_axis).
    # Each column of the matrix represents one of the axes.
    rot_mat = np.stack((x_axis, y_axis, z_axis), axis=1)
    
    # Convert the rotation matrix to a quaternion.
    return quaternion_from_matrix(rot_mat)

def quaternion_from_matrix(matrix):
    """
    Converts a 3x3 rotation matrix to a quaternion.
    
    The conversion checks the trace of the matrix to determine the correct 
    formulation for extracting the quaternion components.
    
    Args:
        matrix (np.array): A 3x3 rotation matrix.
    
    Returns:
        A numpy array representing the quaternion (w, x, y, z).
    """
    m = matrix
    trace = np.trace(m)
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    else:
        # Identify the major diagonal element with the greatest value.
        i = np.argmax(np.diag(m))
        if i == 0:
            s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif i == 1:
            s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    return np.array([w, x, y, z])

def translation_matrix(pos):
    """
    Constructs a 4x4 translation matrix for a given 3D position.
    
    Args:
        pos (np.array): A 3-element array representing the translation vector.
        
    Returns:
        A 4x4 numpy array representing the translation matrix.
    """
    mat = np.eye(4)
    # Set the last column to the translation vector.
    mat[:3, 3] = pos
    return mat

def rotation_matrix_from_quaternion(quat):
    """
    Constructs a 4x4 rotation matrix from a quaternion.
    
    Args:
        quat (np.array): A quaternion in the form (w, x, y, z).
        
    Returns:
        A 4x4 numpy array representing the rotation matrix.
    """
    w, x, y, z = quat
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y, 0],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x, 0],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y, 0],
        [0, 0, 0, 1]
    ])
