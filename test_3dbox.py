#!/usr/bin/env python3

import numpy as np
import json
from pathlib import Path

def analyze_3d_bboxes():
    """读取并分析3D边界框数据"""
    
    # 读取数据
    data_dir = Path('/home/kewei/YING/isaac_to_omni/test')
    bbox_3d = np.load(str(data_dir / 'bounding_box_3d_0000.npy'))
    print(bbox_3d)

    
    with open(data_dir / 'bounding_box_3d_labels_0000.json', 'r') as f:
        bbox_3d_labels = json.load(f)
    
    with open(data_dir / 'bounding_box_3d_prim_paths_0000.json', 'r') as f:
        bbox_3d_prim_paths = json.load(f)
    
    print("=" * 60)
    print("3D边界框数据分析")
    print("=" * 60)
    print(f"\n边界框数组信息:")
    print(f"  形状: {bbox_3d.shape}")
    print(f"  数据类型: {bbox_3d.dtype}")
    print(f"  总大小: {bbox_3d.size}")
    
    print(f"\n标签信息:")
    print(f"  标签数量: {len(bbox_3d_labels)}")
    for key, value in bbox_3d_labels.items():
        print(f"    {key}: {value}")
    
    print(f"\nPrim路径信息:")
    print(f"  路径数量: {len(bbox_3d_prim_paths)}")
    for i, path in enumerate(bbox_3d_prim_paths):
        print(f"    [{i}]: {path}")
    
    # 详细分析每个边界框
    print(f"\n详细边界框信息:")
    print("-" * 60)
    
    output_lines = []
    output_lines.append("# 3D Bounding Boxes Data")
    output_lines.append(f"# Total boxes: {len(bbox_3d)}")
    output_lines.append(f"# Format: Each box has 8 corners (x, y, z)")
    output_lines.append("")
    
    for i in range(len(bbox_3d)):
        bbox = bbox_3d[i]
        
        print(f"\n边界框 [{i}]:")
        print(f"  数据类型: {bbox.dtype}")
        
        # 获取对应的标签和路径
        label_info = bbox_3d_labels.get(str(i), {})
        class_name = label_info.get('class', 'unknown')
        prim_path = bbox_3d_prim_paths[i] if i < len(bbox_3d_prim_paths) else 'unknown'
        
        print(f"  类别: {class_name}")
        print(f"  路径: {prim_path}")
        
        # 添加到输出
        output_lines.append(f"Box {i}: {class_name} ({prim_path})")
        output_lines.append(f"Dtype: {bbox.dtype}")
        
        # 解析结构化数组
        semantic_id = bbox['semanticId']
        x_min = bbox['x_min']
        y_min = bbox['y_min']
        z_min = bbox['z_min']
        x_max = bbox['x_max']
        y_max = bbox['y_max']
        z_max = bbox['z_max']
        transform = bbox['transform']
        occlusion_ratio = bbox['occlusionRatio']
        
        print(f"  语义ID: {semantic_id}")
        print(f"  边界框范围:")
        print(f"    X: [{x_min:.4f}, {x_max:.4f}]")
        print(f"    Y: [{y_min:.4f}, {y_max:.4f}]")
        print(f"    Z: [{z_min:.4f}, {z_max:.4f}]")
        print(f"  遮挡率: {occlusion_ratio:.4f}")
        
        output_lines.append(f"  Semantic ID: {semantic_id}")
        output_lines.append(f"  Bounding Box Range:")
        output_lines.append(f"    X: [{x_min:.4f}, {x_max:.4f}]")
        output_lines.append(f"    Y: [{y_min:.4f}, {y_max:.4f}]")
        output_lines.append(f"    Z: [{z_min:.4f}, {z_max:.4f}]")
        output_lines.append(f"  Occlusion Ratio: {occlusion_ratio:.4f}")
        
        # 计算中心和尺寸
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        center_z = (z_min + z_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        depth = z_max - z_min
        
        print(f"  中心点: ({center_x:.4f}, {center_y:.4f}, {center_z:.4f})")
        print(f"  尺寸 (W×H×D): ({width:.4f}, {height:.4f}, {depth:.4f})")
        
        output_lines.append(f"  Center: ({center_x:.4f}, {center_y:.4f}, {center_z:.4f})")
        output_lines.append(f"  Size (W×H×D): ({width:.4f}, {height:.4f}, {depth:.4f})")
        
        # 输出变换矩阵
        print(f"  变换矩阵 (4x4):")
        output_lines.append("  Transform Matrix (4x4):")
        for row_idx, row in enumerate(transform):
            print(f"    [{row[0]:8.4f}, {row[1]:8.4f}, {row[2]:8.4f}, {row[3]:8.4f}]")
            output_lines.append(f"    [{row[0]:8.4f}, {row[1]:8.4f}, {row[2]:8.4f}, {row[3]:8.4f}]")
        
        # 计算8个角点
        corners_local = np.array([
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max],
        ])
        
        print(f"  8个角点坐标 (局部):")
        output_lines.append("  8 Corner Points (Local):")
        output_lines.append("  Corner    X          Y          Z")
        output_lines.append("  " + "-" * 40)
        for j, corner in enumerate(corners_local):
            print(f"    角点 {j}: ({corner[0]:8.4f}, {corner[1]:8.4f}, {corner[2]:8.4f})")
            output_lines.append(f"  {j}      {corner[0]:8.4f}   {corner[1]:8.4f}   {corner[2]:8.4f}")
        
        output_lines.append("")
    
    # 保存为txt文件
    output_file = data_dir / 'bounding_box_3d_analysis.txt'
    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines))
    
    print("\n" + "=" * 60)
    print(f"分析结果已保存到: {output_file}")
    print("=" * 60)
    
    # 额外保存一个简化的格式（KITTI格式风格）
    kitti_output = []
    kitti_output.append("# Simplified format: class x y z w h d")
    for i in range(len(bbox_3d)):
        bbox = bbox_3d[i]
        label_info = bbox_3d_labels.get(str(i), {})
        class_name = label_info.get('class', 'unknown')
        
        # 从结构化数组中提取数据
        x_min = float(bbox['x_min'])
        y_min = float(bbox['y_min'])
        z_min = float(bbox['z_min'])
        x_max = float(bbox['x_max'])
        y_max = float(bbox['y_max'])
        z_max = float(bbox['z_max'])
        
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        center_z = (z_min + z_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        depth = z_max - z_min
        
        kitti_output.append(
            f"{class_name} {center_x:.4f} {center_y:.4f} {center_z:.4f} "
            f"{width:.4f} {height:.4f} {depth:.4f}"
        )
    
    kitti_file = data_dir / 'bounding_box_3d_simple.txt'
    with open(kitti_file, 'w') as f:
        f.write('\n'.join(kitti_output))
    
    print(f"简化格式已保存到: {kitti_file}")
    
if __name__ == '__main__':
    analyze_3d_bboxes()
