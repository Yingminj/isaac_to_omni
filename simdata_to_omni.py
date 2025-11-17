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
        self.data_dir = Path('/home/kewei/YING/isaac_to_omni/test')
        
        self.get_logger().info('Isaac Sim to ROS2 visualizer initialized')
        
    def load_data(self):
        """Load all required data files"""
        try:
            # Load images
            rgb_img = cv2.imread(str(self.data_dir / 'rgb_0000.png'))
            instance_seg = cv2.imread(str(self.data_dir / 'instance_segmentation_0000.png'), cv2.IMREAD_UNCHANGED)
            depth = np.load(str(self.data_dir / 'distance_to_camera_0000.npy'))
            
            # Load 2D bounding boxes
            bbox_2d = np.load(str(self.data_dir / 'bounding_box_2d_tight_0000.npy'))
            with open(self.data_dir / 'bounding_box_2d_tight_labels_0000.json', 'r') as f:
                bbox_2d_labels = json.load(f)
            with open(self.data_dir / 'bounding_box_2d_tight_prim_paths_0000.json', 'r') as f:
                bbox_2d_prim_paths = json.load(f)
            
            # Load 3D bounding boxes
            bbox_3d = np.load(str(self.data_dir / 'bounding_box_3d_0000.npy'))
            with open(self.data_dir / 'bounding_box_3d_labels_0000.json', 'r') as f:
                bbox_3d_labels = json.load(f)
            with open(self.data_dir / 'bounding_box_3d_prim_paths_0000.json', 'r') as f:
                bbox_3d_prim_paths = json.load(f)
            
            # Load instance segmentation mappings
            with open(self.data_dir / 'instance_segmentation_mapping_0000.json', 'r') as f:
                instance_mapping = json.load(f)
            with open(self.data_dir / 'instance_segmentation_semantics_mapping_0000.json', 'r') as f:
                semantics_mapping = json.load(f)
            
            return {
                'rgb': rgb_img,
                'instance_seg': instance_seg,
                'depth': depth,
                'bbox_2d': bbox_2d,
                'bbox_2d_labels': bbox_2d_labels,
                'bbox_2d_prim_paths': bbox_2d_prim_paths,
                'bbox_3d': bbox_3d,
                'bbox_3d_labels': bbox_3d_labels,
                'bbox_3d_prim_paths': bbox_3d_prim_paths,
                'instance_mapping': instance_mapping,
                'semantics_mapping': semantics_mapping
            }
        except Exception as e:
            self.get_logger().error(f'Error loading data: {str(e)}')
            return None
    
    def depth_to_pointcloud(self, rgb, depth, fx=525.0, fy=525.0, cx=None, cy=None):
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
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        # Filter out invalid points
        valid_mask = (z > 0) & (z < 10.0)  # Filter points with reasonable depth
        
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
    
    def create_bbox_markers(self, bbox_3d, bbox_3d_labels, bbox_3d_prim_paths):
        """Create MarkerArray for 3D bounding boxes"""
        marker_array = MarkerArray()
        
        # Define colors for different classes
        color_map = {
            'nvidia_cube': (1.0, 0.0, 0.0),  # Red
            'chess_box': (0.0, 1.0, 0.0),    # Green
            'mug': (0.0, 0.0, 1.0),          # Blue
        }
        
        for i, bbox in enumerate(bbox_3d):
            if len(bbox) == 0:
                continue
            
            # Get label for this bounding box
            class_name = bbox_3d_labels.get(str(i), {}).get('class', 'unknown')
            color = color_map.get(class_name, (1.0, 1.0, 1.0))
            
            # Create marker for bounding box
            marker = Marker()
            marker.header.frame_id = 'camera_frame'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'bounding_boxes'
            marker.id = i
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            
            # Set scale
            marker.scale.x = 0.02  # Line width
            
            # Set color
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 1.0
            
            # Extract 8 corners from bbox (assuming bbox shape is (8, 3))
            if bbox.shape[0] == 8:
                corners = bbox
                
                # Define edges of the bounding box (12 edges connecting 8 corners)
                edges = [
                    (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
                    (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
                    (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical edges
                ]
                
                from geometry_msgs.msg import Point
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
            if bbox.shape[0] == 8:
                center = np.mean(bbox, axis=0)
                text_marker.pose.position.x = float(center[0])
                text_marker.pose.position.y = float(center[1])
                text_marker.pose.position.z = float(center[2]) + 0.1
            
            text_marker.text = class_name
            text_marker.scale.z = 0.1
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
        
        # Generate point cloud from depth and RGB
        points, colors = self.depth_to_pointcloud(data['rgb'], data['depth'])
        
        # Publish point cloud
        pc_msg = self.create_pointcloud_msg(points, colors)
        self.pointcloud_pub.publish(pc_msg)
        self.get_logger().info(f'Published point cloud with {len(points)} points')
        
        # Publish bounding boxes
        bbox_markers = self.create_bbox_markers(
            data['bbox_3d'],
            data['bbox_3d_labels'],
            data['bbox_3d_prim_paths']
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

