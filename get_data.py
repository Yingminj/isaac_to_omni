import os
import shutil
from pathlib import Path
import re

def get_file_id(filename):
    """从文件名中提取ID"""
    match = re.search(r'(\d{4})', filename)
    return int(match.group(1)) if match else -1

def organize_dataset(base_path, output_path):
    """
    读取Replicator数据并组织成train/test数据集
    每25个数据打包，前3包给train，第4包给test
    """
    base_path = Path(base_path)
    output_path = Path(output_path)
    
    # 定义输入文件夹
    rgb_dir = base_path / "rgb"
    meta_dir = base_path / "meta"
    mask_dir = base_path / "mask"
    depth_dir = base_path / "depth"
    
    # 检查文件夹是否存在
    for dir_path in [rgb_dir, meta_dir, mask_dir, depth_dir]:
        if not dir_path.exists():
            print(f"警告: {dir_path} 不存在")
            return
    
    # 获取所有RGB文件并按ID排序
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')],
                       key=get_file_id)
    
    print(f"找到 {len(rgb_files)} 个数据文件")
    
    # 创建输出目录
    train_dir = output_path / "train"
    test_dir = output_path / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理每个数据包
    batch_size = 25
    total_files = len(rgb_files)
    batch_count = 0
    
    for i in range(0, total_files, batch_size):
        batch_files = rgb_files[i:i + batch_size]
        batch_count += 1
        
        # 确定这个批次是train还是test (前3个batch给train,第4个给test)
        is_train = (batch_count % 4) != 0
        target_dir = train_dir if is_train else test_dir
        dataset_type = "train" if is_train else "test"
        
        print(f"处理批次 {batch_count} ({len(batch_files)} 个文件) -> {dataset_type}")
        
        # 为每个批次创建子目录
        batch_dir = target_dir / f"{batch_count:04d}"
        
        for dir_path in [batch_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 复制文件
        for rgb_file in batch_files:
            file_id = get_file_id(rgb_file)
            
            # 构造对应的文件名
            # rgb_file = f"{file_id:04d}_color.png"
            meta_file = f"{file_id:04d}_meta.json"
            mask_file = f"{file_id:04d}_mask.png"
            depth_file = f"{file_id:04d}_depth.exr"
            # 将rgb_file重命名后保存
            rgb_file_renamed = f"{file_id:04d}_color.png"
            
            # 复制文件
            try:
                shutil.copy2(rgb_dir / rgb_file, batch_dir / rgb_file_renamed)
                shutil.copy2(meta_dir / meta_file, batch_dir / meta_file)
                shutil.copy2(mask_dir / mask_file, batch_dir / mask_file)
                shutil.copy2(depth_dir / depth_file, batch_dir / depth_file)
            except FileNotFoundError as e:
                print(f"警告: 文件缺失 - {e}")
    
    print(f"\n完成! 总共处理 {batch_count} 个批次")
    print(f"Train批次: {sum(1 for i in range(1, batch_count + 1) if i % 4 != 0)}")
    print(f"Test批次: {sum(1 for i in range(1, batch_count + 1) if i % 4 == 0)}")

if __name__ == "__main__":
    # 设置路径
    base_path = "datasets/basic1117-2/Replicator_03"
    output_path = "datasets/processed"
    
    organize_dataset(base_path, output_path)
