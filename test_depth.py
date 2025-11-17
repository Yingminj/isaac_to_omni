#!/usr/bin/env python3

import numpy as np
import cv2
from pathlib import Path

def visualize_depth():
    """读取并可视化深度图"""
    # 读取深度数据
    depth_path = Path('/home/kewei/YING/isaac_to_omni/test/distance_to_camera_0000.npy')
    depth = np.load(str(depth_path))
    
    print(f"深度图信息:")
    print(f"  形状: {depth.shape}")
    print(f"  数据类型: {depth.dtype}")
    print(f"  最小值: {np.min(depth)}")
    print(f"  最大值: {np.max(depth)}")
    print(f"  平均值: {np.mean(depth)}")
    print(f"  包含 inf 的数量: {np.isinf(depth).sum()}")
    print(f"  包含 nan 的数量: {np.isnan(depth).sum()}")
    
    # 处理无效值
    depth_clean = depth.copy()
    depth_clean[~np.isfinite(depth_clean)] = 0
    
    # 方法1: 归一化到0-255用于可视化（灰度图）
    depth_valid = depth_clean[depth_clean > 0]
    if len(depth_valid) > 0:
        min_val = np.min(depth_valid)
        max_val = np.max(depth_valid)
        print(f"  清理后最小值: {min_val}")
        print(f"  清理后最大值: {max_val}")
        depth_normalized = np.zeros_like(depth_clean)
        valid_mask = depth_clean > 0
        depth_normalized[valid_mask] = ((depth_clean[valid_mask] - min_val) / (max_val - min_val) * 255)
        depth_gray = depth_normalized.astype(np.uint8)
    else:
        depth_gray = np.zeros(depth_clean.shape, dtype=np.uint8)
    
    # 方法2: 使用伪彩色映射（热力图）
    depth_color = cv2.applyColorMap(depth_gray, cv2.COLORMAP_JET)
    
    # 方法3: 反转颜色（近的物体更亮）
    depth_inverted = 255 - depth_gray
    depth_inverted_color = cv2.applyColorMap(depth_inverted, cv2.COLORMAP_JET)
    
    # 显示原始RGB图像（如果存在）
    rgb_path = Path('/home/kewei/YING/isaac_to_omni/test/rgb_0000.png')
    if rgb_path.exists():
        rgb = cv2.imread(str(rgb_path))
        cv2.imshow('RGB Image', rgb)
    
    # 显示所有可视化结果
    cv2.imshow('Depth - Grayscale', depth_gray)
    cv2.imshow('Depth - Heatmap (JET)', depth_color)
    cv2.imshow('Depth - Inverted Heatmap', depth_inverted_color)
    
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 保存可视化结果
    output_dir = Path('/home/kewei/YING/isaac_to_omni/test')
    cv2.imwrite(str(output_dir / 'depth_gray.png'), depth_gray)
    cv2.imwrite(str(output_dir / 'depth_heatmap.png'), depth_color)
    cv2.imwrite(str(output_dir / 'depth_inverted.png'), depth_inverted_color)
    print(f"\n可视化结果已保存到 {output_dir}")

if __name__ == '__main__':
    visualize_depth()
