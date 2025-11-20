#!/usr/bin/env python3

import numpy as np
import json
from pathlib import Path
from scipy.spatial.transform import Rotation as R

def calculate_camera_intrinsics(camera_params):
    """Calculate camera intrinsics from Isaac Sim camera parameters"""
    sensor_w_mm, sensor_h_mm = camera_params["cameraAperture"]
    f_mm = camera_params["cameraFocalLength"]
    width, height = camera_params["renderProductResolution"]
    aperture_offset_x_mm, aperture_offset_y_mm = camera_params["cameraApertureOffset"]
    
    # Calculate pixel focal lengths and principal point
    fx = f_mm * (width / sensor_w_mm)
    fy = f_mm * (height / sensor_h_mm)
    cx = width / 2.0 + aperture_offset_x_mm * (width / sensor_w_mm)
    cy = height / 2.0 + aperture_offset_y_mm * (height / sensor_h_mm)
    
    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": int(width),
        "height": int(height)
    }

def transform_bbox_to_camera(bbox, cam_view_matrix):
    """Transform bounding box from local to camera coordinates"""
    # Rotation matrix: 180 degrees around X-axis (Isaac to ROS camera convention)
    rot_x_180 = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    
    # Extract bbox transform and transpose (row-major to column-major)
    transform = bbox['transform'].T
    
    # Combine transformations
    combined_transform = rot_x_180 @ cam_view_matrix @ transform
    
    # Extract rotation (3x3) and translation (3x1)
    rotation_matrix = combined_transform[:3, :3]
    translation = combined_transform[:3, 3]
    
    # Calculate bounding box dimensions
    x_min, y_min, z_min = float(bbox['x_min']), float(bbox['y_min']), float(bbox['z_min'])
    x_max, y_max, z_max = float(bbox['x_max']), float(bbox['y_max']), float(bbox['z_max'])
    bbox_side_len = [x_max - x_min, y_max - y_min, z_max - z_min]
    
    return rotation_matrix, translation, bbox_side_len

def rotation_matrix_to_quaternion(rotation_matrix):
    """Convert rotation matrix to quaternion (w, x, y, z)"""
    rot = R.from_matrix(rotation_matrix)
    quat_xyzw = rot.as_quat()  # Returns [x, y, z, w]
    return [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]  # Convert to [w, x, y, z]

def rotation_matrix_to_euler(rotation_matrix):
    """Convert rotation matrix to euler angles (xyz)"""
    rot = R.from_matrix(rotation_matrix)
    euler = rot.as_euler('xyz', degrees=False)
    return euler.tolist()

def process_frame(data_dir, frame_id):
    """Process a single frame and generate meta JSON"""
    # Load camera parameters
    camera_params_path = data_dir / "camera_params" / f"camera_params_{frame_id:04d}.json"
    with open(camera_params_path, 'r') as f:
        camera_params = json.load(f)
    
    # Calculate intrinsics
    intrinsics = calculate_camera_intrinsics(camera_params)
    
    # Parse camera view transform matrix
    cam_view_matrix = np.array(camera_params['cameraViewTransform']).reshape(4, 4).T
    
    # Load 3D bounding boxes
    bbox_3d_path = data_dir / "bounding_box_3d" / f"bounding_box_3d_{frame_id:04d}.npy"
    bbox_3d = np.load(bbox_3d_path)
    
    # Load 3D labels
    labels_3d_path = data_dir / "bounding_box_3d" / f"bounding_box_3d_labels_{frame_id:04d}.json"
    with open(labels_3d_path, 'r') as f:
        bbox_3d_labels = json.load(f)
    
    # Load 2D labels for class_label mapping
    labels_2d_path = data_dir / "bounding_box_2d_tight" / f"bounding_box_2d_tight_labels_{frame_id:04d}.json"
    with open(labels_2d_path, 'r') as f:
        bbox_2d_labels = json.load(f)
    
    # Load prim paths
    prim_paths_path = data_dir / "bounding_box_3d" / f"bounding_box_3d_prim_paths_{frame_id:04d}.json"
    with open(prim_paths_path, 'r') as f:
        prim_paths = json.load(f)
    
    # Build output structure
    output = {
        "camera": {
            "intrinsics": intrinsics,
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "translation": [0.0, 0.0, 0.0],
            "scene_obj_path": "",
            "background_image_path": "",
            "background_depth_path": "",
            "distances": [],
            "kind": ""
        },
        "scene_dataset": "own_real",
        "env_param": {},
        "face_up": True,
        "concentrated": False,
        "comments": "sim workpiece data",
        "runtime_seed": -1,
        "baseline_dis": 0,
        "emitter_dist_l": 0,
        "objects": {}
    }
    
    # Process each bounding box
    for i, bbox in enumerate(bbox_3d):
        semantic_id = str(bbox['semanticId'])
        
        # Get class name from 3D labels
        if semantic_id not in bbox_3d_labels:
            continue
        
        class_info = bbox_3d_labels[semantic_id].get('class', 'unknown')
        
        # Get class_label from 2D labels (use semantic_id as key)
        class_label = int(semantic_id) if semantic_id in bbox_2d_labels else i + 1
        
        prim_path = prim_paths[i] if i < len(prim_paths) else "unknown"
        
        # Skip invalid bounding boxes
        x_min, y_min, z_min = float(bbox['x_min']), float(bbox['y_min']), float(bbox['z_min'])
        x_max, y_max, z_max = float(bbox['x_max']), float(bbox['y_max']), float(bbox['z_max'])
        
        if abs(x_max - x_min) < 1e-6 or abs(y_max - y_min) < 1e-6 or abs(z_max - z_min) < 1e-6:
            continue
        
        # Transform to camera coordinates
        rotation_matrix, translation, bbox_side_len = transform_bbox_to_camera(bbox, cam_view_matrix)
        
        # Convert rotation to quaternion and euler
        quaternion = rotation_matrix_to_quaternion(rotation_matrix)
        # euler_angles = rotation_matrix_to_euler(rotation_matrix)
        
        # Create object ID and name using class_label
        object_id = f"{class_label}_{class_info}_{frame_id:03d}"
        
        # Build object entry
        output["objects"][object_id] = {
            "id": i + 1,
            "meta": {               
                "class_name": class_info,
                "class_label": class_label,
                "instance_path": prim_path,
                "scale": [1.0, 1.0, 1.0],                
                "bbox_side_len": bbox_side_len,
                "is_background": False,
                "oid": f"sim-{class_info}_{frame_id:03d}",
            },
            "quaternion_wxyz": quaternion,
            "translation": translation.tolist(),
            "world_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "world_translation": [0.0, 0.0, 0.0],
            "is_valid": True,           
            "material": [],
            # "rotation": euler_angles
        }
    
    return output

def main():
    # Set data directory
    data_dir = Path('/home/kewei/YING/isaac_to_omni/datasets/basic1117-2/Replicator_03')
    output_dir = data_dir / "meta"
    output_dir.mkdir(exist_ok=True)
    
    # Find all camera params files to determine frame range
    camera_params_dir = data_dir / "camera_params"
    camera_params_files = sorted(camera_params_dir.glob("camera_params_*.json"))
    
    print(f"Found {len(camera_params_files)} frames to process")
    
    # Process each frame
    for camera_params_file in camera_params_files:
        # Extract frame ID from filename
        frame_id = int(camera_params_file.stem.split('_')[-1])
        
        try:
            print(f"Processing frame {frame_id:04d}...")
            output = process_frame(data_dir, frame_id)
            
            # Save output JSON
            output_path = output_dir / f"{frame_id:04d}_meta.json"
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2)
            
            print(f"  Saved to {output_path}")
            print(f"  Found {len(output['objects'])} objects")
            
        except Exception as e:
            print(f"  Error processing frame {frame_id:04d}: {str(e)}")
            continue
    
    print("Processing complete!")

if __name__ == '__main__':
    main()
