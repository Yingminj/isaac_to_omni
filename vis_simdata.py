#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
import numpy as np
import json
import cv2
from pathlib import Path
import struct

class IsaacSimToROS2(Node):
    def __init__(self):
        super().__init__('isaac_sim_visualizer')
        
        # Publishers
        self.pointcloud_pub = self.create_publisher(PointCloud2, '/pointcloud', 10)
        self.bbox_pub = self.create_publisher(MarkerArray, '/bounding_boxes', 10)
        
        # Timer for publishing
        self.timer = self.create_timer(1.0, self.publish_data)
        
        # Data directory
        self.data_dir = Path('/home/kewei/YING/isaac_to_omni/datasets/basic1117/Replicator_02')
        
        self.get_logger().info('Isaac Sim to ROS2 visualizer initialized')
        
    def load_data(self):
        """Load all required data files"""
        try:
            # Load images
            id = 15
            # old
            # rgb_img = cv2.imread(str(self.data_dir/ f'rgb_{id:04d}.png'))
            # # instance_seg = cv2.imread(str(self.data_dir / f'instance_segmentation_{id:04d}.png'), cv2.IMREAD_UNCHANGED)
            # depth = np.load(str(self.data_dir/ f'distance_to_camera_{id:04d}.npy'))
            
            # # Load 2D bounding boxes
            # bbox_2d = np.load(str(self.data_dir / f'bounding_box_2d_tight_{id:04d}.npy'))
            # with open(self.data_dir/ f'bounding_box_2d_tight_labels_{id:04d}.json', 'r') as f:
            #     bbox_2d_labels = json.load(f)
            # with open(self.data_dir / f'bounding_box_2d_tight_prim_paths_{id:04d}.json', 'r') as f:
            #     bbox_2d_prim_paths = json.load(f)
            
            # # Load 3D bounding boxes
            # bbox_3d = np.load(str(self.data_dir/ f'bounding_box_3d_{id:04d}.npy'))
            # with open(self.data_dir  / f'bounding_box_3d_labels_{id:04d}.json', 'r') as f:
            #     bbox_3d_labels = json.load(f)
            # with open(self.data_dir / f'bounding_box_3d_prim_paths_{id:04d}.json', 'r') as f:
            #     bbox_3d_prim_paths = json.load(f)
            
            # # Load camera parameters
            # with open(self.data_dir/ f'camera_params_{id:04d}.json', 'r') as f:
            #     camera_params = json.load(f) 

            #new
            rgb_img = cv2.imread(str(self.data_dir / "rgb" / f'rgb_{id:04d}.png'))
            # instance_seg = cv2.imread(str(self.data_dir / f'instance_segmentation_{id:04d}.png'), cv2.IMREAD_UNCHANGED)
            # depth = np.load(str(self.data_dir / "distance_to_camera" / f'distance_to_camera_{id:04d}.npy'))
            depth = np.load(str(self.data_dir / "distance_to_image_plane" / f'distance_to_image_plane_{id:04d}.npy'))
            # basic1117/Replicator_02/distance_to_image_plane/distance_to_image_plane_0000.npy
            
            # Load 2D bounding boxes
            bbox_2d = np.load(str(self.data_dir / "bounding_box_2d_tight" / f'bounding_box_2d_tight_{id:04d}.npy'))
            with open(self.data_dir/ "bounding_box_2d_tight" / f'bounding_box_2d_tight_labels_{id:04d}.json', 'r') as f:
                bbox_2d_labels = json.load(f)
            with open(self.data_dir / "bounding_box_2d_tight" / f'bounding_box_2d_tight_prim_paths_{id:04d}.json', 'r') as f:
                bbox_2d_prim_paths = json.load(f)
            
            # Load 3D bounding boxes
            bbox_3d = np.load(str(self.data_dir/ "bounding_box_3d" / f'bounding_box_3d_{id:04d}.npy'))
            with open(self.data_dir / "bounding_box_3d" / f'bounding_box_3d_labels_{id:04d}.json', 'r') as f:
                bbox_3d_labels = json.load(f)
            with open(self.data_dir / "bounding_box_3d" / f'bounding_box_3d_prim_paths_{id:04d}.json', 'r') as f:
                bbox_3d_prim_paths = json.load(f)
            
            # Load camera parameters
            with open(self.data_dir/ "camera_params" / f'camera_params_{id:04d}.json', 'r') as f:
                camera_params = json.load(f)            
            
            # Load instance segmentation mappings
            # with open(self.data_dir / f'instance_segmentation_mapping_{id:04d}.json', 'r') as f:
            #     instance_mapping = json.load(f)
            # with open(self.data_dir / f'instance_segmentation_semantics_mapping_{id:04d}.json', 'r') as f:
            #     semantics_mapping = json.load(f)
            
            return {
                'rgb': rgb_img,
                # 'instance_seg': instance_seg,
                'depth': depth,
                'bbox_2d': bbox_2d,
                'bbox_2d_labels': bbox_2d_labels,
                'bbox_2d_prim_paths': bbox_2d_prim_paths,
                'bbox_3d': bbox_3d,
                'bbox_3d_labels': bbox_3d_labels,
                'bbox_3d_prim_paths': bbox_3d_prim_paths,
                'camera_params': camera_params,
                # 'instance_mapping': instance_mapping,
                # 'semantics_mapping': semantics_mapping
            }
        except Exception as e:
            self.get_logger().error(f'Error loading data: {str(e)}')
            return None
    
    def depth_to_pointcloud(self, rgb, depth, fx=400.0, fy=400.0, cx=None, cy=None):
        """Convert depth image to point cloud with color"""
        h, w = depth.shape
        if cx is None:
            cx = w / 2.0
        if cy is None:
            cy = h / 2.0
        
        # Create mesh grid
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Convert to 3D points
        z = depth
        # Replace invalid values (inf, nan) with 0
        z = np.where(np.isfinite(z), z, 0)
        
        # Filter out invalid points BEFORE calculation
        valid_mask = (z > 0) & (z < 10000.0)  # Filter points with reasonable depth
        
        # Only calculate for valid points to avoid warnings
        x = np.zeros_like(z, dtype=np.float32)
        y = np.zeros_like(z, dtype=np.float32)
        x[valid_mask] = (u[valid_mask] - cx) * z[valid_mask] / fx
        y[valid_mask] = (v[valid_mask] - cy) * z[valid_mask] / fy
        
        points = np.stack([x, y, z], axis=-1)
        points = points[valid_mask]
        
        # Get RGB colors
        if rgb is not None and rgb.shape[:2] == (h, w):
            colors = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            colors = colors[valid_mask]
        else:
            colors = np.ones((points.shape[0], 3), dtype=np.uint8) * 128
        
        return points, colors
    
    def create_pointcloud_msg(self, points, colors):
        """Create PointCloud2 message"""
        header = Header()
        header.frame_id = 'camera_frame'
        header.stamp = self.get_clock().now().to_msg()
        
        # Define point cloud fields
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]
        
        # Pack point cloud data
        cloud_data = []
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = colors[i]
            rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, 255))[0]
            cloud_data.append(struct.pack('fffI', x, y, z, rgb))
        
        # Create PointCloud2 message
        pc2_msg = PointCloud2()
        pc2_msg.header = header
        pc2_msg.height = 1
        pc2_msg.width = len(points)
        pc2_msg.fields = fields
        pc2_msg.is_bigendian = False
        pc2_msg.point_step = 16
        pc2_msg.row_step = pc2_msg.point_step * pc2_msg.width
        pc2_msg.is_dense = True
        pc2_msg.data = b''.join(cloud_data)
        
        return pc2_msg
    
    def create_bbox_markers(self, bbox_3d, bbox_3d_labels, bbox_3d_prim_paths, camera_view_transform):
        """Create MarkerArray for 3D bounding boxes from structured array"""
        marker_array = MarkerArray()
        from geometry_msgs.msg import Point
        
        # Parse camera view transform matrix (4x4, row-major order from Isaac Sim)
        cam_matrix = np.zeros((4, 4))
        cam_view_matrix = np.array(camera_view_transform).reshape(4, 4)
        cam_view_matrix = cam_view_matrix.T  # Transpose to convert to column-major order
        # cam_tran = cam_view_matrix[:3, 3] 
        # cam_rot = cam_view_matrix[:3, :3]
        # cam_tran = np.linalg.inv(cam_view_matrix)[:3, 3]
        # cam_rot = np.linalg.inv(cam_view_matrix)[:3, :3]
        
        # cam_matrix = np.hstack([cam_rot, cam_tran.reshape(3, 1)])
        # cam_matrix = np.vstack([cam_matrix, [0, 0, 0, 1]])
        # cam_view_matrix = cam_matrix
        rot_x_180 = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        

        self.get_logger().info(f'Camera view transform matrix:\n{cam_view_matrix}')
        
        # Define colors for different classes
        color_map = {
            'nvidia_cube': (1.0, 0.0, 0.0),  # Red
            'chess_box': (0.0, 1.0, 0.0),    # Green
            'mug': (0.0, 0.0, 1.0),          # Blue
        }
        
        for i, bbox in enumerate(bbox_3d):
            # Get label for this bounding box
            id = bbox['semanticId']
            class_name = bbox_3d_labels.get(str(id), {}).get('class', 'unknown')
            
            # Skip unknown classes or check if bbox is valid
            if class_name == 'unknown':
                self.get_logger().info(f'Skipping unknown class at index {i}')
                continue
            
            color = color_map.get(class_name, (1.0, 1.0, 1.0))
            
            # Extract bounding box parameters from structured array
            x_min = float(bbox['x_min'])
            y_min = float(bbox['y_min'])
            z_min = float(bbox['z_min'])
            x_max = float(bbox['x_max'])
            y_max = float(bbox['y_max'])
            z_max = float(bbox['z_max'])
            transform = bbox['transform']
            transform = transform.T
            # transform = np.linalg.inv(transform)  # Invert to get local to world

            # Skip invalid bounding boxes (all zeros or invalid dimensions)
            if abs(x_max - x_min) < 1e-6 or abs(y_max - y_min) < 1e-6 or abs(z_max - z_min) < 1e-6:
                self.get_logger().warn(f'Skipping invalid bbox {i}: dimensions too small')
                continue
            
            # Calculate 8 corners of the bounding box in local coordinates
            corners_local = np.array([
                [x_min, y_min, z_min],  # 0: bottom-front-left
                [x_max, y_min, z_min],  # 1: bottom-front-right
                [x_max, y_max, z_min],  # 2: bottom-back-right
                [x_min, y_max, z_min],  # 3: bottom-back-left
                [x_min, y_min, z_max],  # 4: top-front-left
                [x_max, y_min, z_max],  # 5: top-front-right
                [x_max, y_max, z_max],  # 6: top-back-right
                [x_min, y_max, z_max],  # 7: top-back-left
            ])
            
            # Extract rotation (3x3) and translation (3x1) from transform matrix
            # rotation = transform[:3, :3]  # Top-left 3x3 rotation matrix
            # translation = transform[:3, 3]  # Bottom row, first 3 elements (tx, ty, tz)

            # corners_world = np.zeros((8, 3))
            # corners_world = (rotation @ corners_local.T).T + translation
            
            # corners_world_homogeneous = np.hstack([corners_world, np.ones((8, 1))])
            
            # corners_camera_homogeneous = (cam_view_matrix @ corners_world_homogeneous.T).T

            # corners_camera_homogeneous = (rot_x_180 @ corners_camera_homogeneous.T).T
            
            # corners_camera = corners_camera_homogeneous[:, :3]
            
            # Combine transformations: final_transform = rot_x_180 @ cam_view_matrix @ transform
            # This transforms from local bbox coordinates -> world -> camera -> ROS camera convention
            combined_transform = rot_x_180 @ cam_view_matrix @ transform
            
            # Convert local corners to homogeneous coordinates
            corners_local_homogeneous = np.hstack([corners_local, np.ones((8, 1))])
            
            # Apply combined transform to all corners at once
            corners_camera_homogeneous = (combined_transform @ corners_local_homogeneous.T).T
            
            # Convert back to 3D coordinates (drop the homogeneous coordinate)
            corners_camera = corners_camera_homogeneous[:, :3]

            self.get_logger().info(
                f'Box {i} ({class_name}): '
                # f'World center: ({np.mean(corners_world, axis=0)}), '
                f'Camera center: ({np.mean(corners_camera, axis=0)})'
            )
            
            corners = corners_camera
            
            # Create marker for bounding box
            marker = Marker()
            marker.header.frame_id = 'camera_frame'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'bounding_boxes'
            marker.id = i
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            
            # Set scale (line width)
            marker.scale.x = 0.005
            
            # Set color
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 1.0
            
            # Define edges of the bounding box (12 edges connecting 8 corners)
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
                (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
                (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical edges
            ]
            
            # Add edge points
            for edge in edges:
                p1 = Point()
                p1.x, p1.y, p1.z = float(corners[edge[0]][0]), float(corners[edge[0]][1]), float(corners[edge[0]][2])
                marker.points.append(p1)
                
                p2 = Point()
                p2.x, p2.y, p2.z = float(corners[edge[1]][0]), float(corners[edge[1]][1]), float(corners[edge[1]][2])
                marker.points.append(p2)
            
            marker_array.markers.append(marker)
            
            # Add text label
            text_marker = Marker()
            text_marker.header.frame_id = 'camera_frame'
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.ns = 'bbox_labels'
            text_marker.id = i + 1000
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            # Position at center of bbox
            center = np.mean(corners, axis=0)
            text_marker.pose.position.x = float(center[0])
            text_marker.pose.position.y = float(center[1])
            text_marker.pose.position.z = float(center[2]) + 0.05
            
            text_marker.text = class_name
            text_marker.scale.z = 0.02
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            
            marker_array.markers.append(text_marker)
        
        return marker_array
    
    def publish_data(self):
        """Load and publish all data"""
        data = self.load_data()
        if data is None:
            return
        
        sensor_w_mm, sensor_h_mm = data['camera_params']["cameraAperture"]
        f_mm = data['camera_params']["cameraFocalLength"]
        width, height = data['camera_params']["renderProductResolution"]
        aperture_offset_x_mm, aperture_offset_y_mm = data['camera_params']["cameraApertureOffset"]

        # 像素焦距与主点（像素）
        fx = f_mm * (width  / sensor_w_mm)
        fy = f_mm * (height / sensor_h_mm)
        cx = width  / 2.0 + aperture_offset_x_mm * (width  / sensor_w_mm)
        cy = height / 2.0 + aperture_offset_y_mm * (height / sensor_h_mm)   

        # Generate point cloud from depth and RGB
        points, colors = self.depth_to_pointcloud(data['rgb'], data['depth'], fx, fy, cx, cy)
        
        # Publish point cloud
        pc_msg = self.create_pointcloud_msg(points, colors)
        self.pointcloud_pub.publish(pc_msg)
        self.get_logger().info(f'Published point cloud with {len(points)} points')
        
        # Publish bounding boxes with camera transform
        bbox_markers = self.create_bbox_markers(
            data['bbox_3d'],
            data['bbox_3d_labels'],
            data['bbox_3d_prim_paths'],
            data['camera_params']['cameraViewTransform']
        )
        self.bbox_pub.publish(bbox_markers)
        self.get_logger().info(f'Published {len(bbox_markers.markers)} markers')

def main(args=None):
    rclpy.init(args=args)
    node = IsaacSimToROS2()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

