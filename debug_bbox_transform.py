#!/usr/bin/env python3

import numpy as np
import json
from pathlib import Path

def debug_bbox_transform():
    """调试3D边界框的变换矩阵"""
    
    # 读取数据
    data_dir = Path('/home/kewei/YING/isaac_to_omni/basic_1104')
    bbox_3d = np.load(str(data_dir / 'bounding_box_3d_0013.npy'))
    
    with open(data_dir / 'bounding_box_3d_labels_0013.json', 'r') as f:
        bbox_3d_labels = json.load(f)
    
    print("=" * 80)
    print("3D边界框变换矩阵调试")
    print("=" * 80)
    
    for i, bbox in enumerate(bbox_3d):
        # 获取标签
        class_name = bbox_3d_labels.get(str(i), {}).get('class', 'unknown')
        
        print(f"\n边界框 [{i}] - {class_name}")
        print("-" * 80)
        
        # 提取数据
        x_min = float(bbox['x_min'])
        y_min = float(bbox['y_min'])
        z_min = float(bbox['z_min'])
        x_max = float(bbox['x_max'])
        y_max = float(bbox['y_max'])
        z_max = float(bbox['z_max'])
        transform = bbox['transform']
        
        print(f"局部边界框范围:")
        print(f"  X: [{x_min:.6f}, {x_max:.6f}]")
        print(f"  Y: [{y_min:.6f}, {y_max:.6f}]")
        print(f"  Z: [{z_min:.6f}, {z_max:.6f}]")
        
        # 计算局部中心和尺寸
        center_local = np.array([
            (x_min + x_max) / 2,
            (y_min + y_max) / 2,
            (z_min + z_max) / 2
        ])
        size = np.array([
            x_max - x_min,
            y_max - y_min,
            z_max - z_min
        ])
        
        print(f"\n局部中心: ({center_local[0]:.6f}, {center_local[1]:.6f}, {center_local[2]:.6f})")
        print(f"尺寸: ({size[0]:.6f}, {size[1]:.6f}, {size[2]:.6f})")
        
        # 打印变换矩阵
        print(f"\n变换矩阵 (4x4):")
        for row in transform:
            print(f"  [{row[0]:10.6f}, {row[1]:10.6f}, {row[2]:10.6f}, {row[3]:10.6f}]")
        
        # 提取平移和旋转
        translation = transform[3, :3]
        rotation = transform[:3, :3]
        
        print(f"\n平移向量: ({translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f})")
        print(f"旋转矩阵:")
        for row in rotation:
            print(f"  [{row[0]:10.6f}, {row[1]:10.6f}, {row[2]:10.6f}]")
        
        # 计算8个角点（局部坐标）
        corners_local = np.array([
            [x_min, y_min, z_min, 1.0],
            [x_max, y_min, z_min, 1.0],
            [x_max, y_max, z_min, 1.0],
            [x_min, y_max, z_min, 1.0],
            [x_min, y_min, z_max, 1.0],
            [x_max, y_min, z_max, 1.0],
            [x_max, y_max, z_max, 1.0],
            [x_min, y_max, z_max, 1.0],
        ])
        
        # 应用变换矩阵
        corners_world = np.zeros((8, 3))
        for j in range(8):
            transformed = transform @ corners_local[j]
            corners_world[j] = transformed[:3]
        
        print(f"\n变换后的8个角点（世界坐标）:")
        print(f"  Corner    X          Y          Z")
        print(f"  " + "-" * 50)
        for j, corner in enumerate(corners_world):
            print(f"  {j}      {corner[0]:10.6f} {corner[1]:10.6f} {corner[2]:10.6f}")
        
        # 计算世界坐标中心
        center_world = np.mean(corners_world, axis=0)
        print(f"\n世界坐标中心: ({center_world[0]:.6f}, {center_world[1]:.6f}, {center_world[2]:.6f})")
        
        # 计算世界坐标边界
        min_world = np.min(corners_world, axis=0)
        max_world = np.max(corners_world, axis=0)
        print(f"世界坐标边界:")
        print(f"  X: [{min_world[0]:.6f}, {max_world[0]:.6f}]")
        print(f"  Y: [{min_world[1]:.6f}, {max_world[1]:.6f}]")
        print(f"  Z: [{min_world[2]:.6f}, {max_world[2]:.6f}]")

if __name__ == '__main__':
    debug_bbox_transform()
