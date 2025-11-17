#!/usr/bin/env python3

import numpy as np
import cv2
import json
from pathlib import Path

def visualize_2d_bboxes():
    """将2D边界框绘制在RGB图像上"""
    
    # 数据目录
    data_dir = Path('/home/kewei/YING/isaac_to_omni/test')
    
    # 读取RGB图像
    rgb_img = cv2.imread(str(data_dir / 'rgb_0000.png'))
    
    # 读取2D边界框
    bbox_2d = np.load(str(data_dir / 'bounding_box_2d_tight_0000.npy'))
    
    # 读取标签
    with open(data_dir / 'bounding_box_2d_tight_labels_0000.json', 'r') as f:
        bbox_labels = json.load(f)
    
    # 读取prim路径（可选）
    with open(data_dir / 'bounding_box_2d_tight_prim_paths_0000.json', 'r') as f:
        bbox_prim_paths = json.load(f)
    
    print("=" * 60)
    print("2D边界框可视化")
    print("=" * 60)
    print(f"图像尺寸: {rgb_img.shape}")
    print(f"边界框数量: {len(bbox_2d)}")
    print(f"边界框数据类型: {bbox_2d.dtype}")
    print()
    
    # 定义颜色映射（不同类别不同颜色）
    color_map = {
        'nvidia_cube': (0, 0, 255),    # 红色 (BGR)
        'chess_box': (0, 255, 0),      # 绿色
        'mug': (255, 0, 0),            # 蓝色
        'unknown': (128, 128, 128),    # 灰色
    }
    
    # 创建输出图像
    output_img = rgb_img.copy()
    
    # 遍历每个边界框
    for i, bbox in enumerate(bbox_2d):
        # 提取边界框数据
        # 格式: (class_id, xmin, ymin, xmax, ymax, occlusion_ratio)
        class_id = int(bbox[0])
        xmin = int(bbox[1])
        ymin = int(bbox[2])
        xmax = int(bbox[3])
        ymax = int(bbox[4])
        occlusion = float(bbox[5])
        
        # 获取类别名称
        class_name = bbox_labels.get(str(class_id), {}).get('class', 'unknown')
        prim_path = bbox_prim_paths[i] if i < len(bbox_prim_paths) else 'unknown'
        
        # 获取颜色
        color = color_map.get(class_name, (255, 255, 255))
        
        print(f"边界框 [{i}]:")
        print(f"  类别ID: {class_id} -> {class_name}")
        print(f"  路径: {prim_path}")
        print(f"  坐标: ({xmin}, {ymin}) -> ({xmax}, {ymax})")
        print(f"  遮挡率: {occlusion:.4f}")
        print()
        
        # 绘制边界框
        cv2.rectangle(output_img, (xmin, ymin), (xmax, ymax), color, 2)
        
        # 准备标签文本
        label_text = f"{class_name} ({occlusion:.2f})"
        
        # 计算文本大小
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, font, font_scale, thickness
        )
        
        # 绘制标签背景
        label_ymin = max(ymin - text_height - 10, 0)
        cv2.rectangle(
            output_img,
            (xmin, label_ymin),
            (xmin + text_width + 10, ymin),
            color,
            -1  # 填充
        )
        
        # 绘制标签文本
        cv2.putText(
            output_img,
            label_text,
            (xmin + 5, ymin - 5),
            font,
            font_scale,
            (255, 255, 255),  # 白色文字
            thickness,
            cv2.LINE_AA
        )
    
    # 显示图像
    cv2.imshow('Original RGB', rgb_img)
    cv2.imshow('2D Bounding Boxes', output_img)
    
    # 保存结果
    output_path = data_dir / 'rgb_0000_with_2d_bboxes.png'
    cv2.imwrite(str(output_path), output_img)
    print(f"结果已保存到: {output_path}")
    
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 创建图例
    legend_height = 150
    legend_width = 300
    legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255
    
    y_offset = 30
    for class_name, color in color_map.items():
        if class_name != 'unknown':
            # 绘制颜色框
            cv2.rectangle(legend, (10, y_offset - 15), (30, y_offset), color, -1)
            cv2.rectangle(legend, (10, y_offset - 15), (30, y_offset), (0, 0, 0), 1)
            
            # 绘制文本
            cv2.putText(
                legend,
                class_name,
                (40, y_offset - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )
            y_offset += 30
    
    cv2.imshow('Legend', legend)
    legend_path = data_dir / 'bbox_legend.png'
    cv2.imwrite(str(legend_path), legend)
    
    print(f"图例已保存到: {legend_path}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    visualize_2d_bboxes()
