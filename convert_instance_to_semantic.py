import cv2
import json
import numpy as np
from pathlib import Path
import re

# Define paths
base_path = Path("datasets/basic1117-2/Replicator_03")
instance_seg_dir = base_path / "instance_segmentation"
output_dir = base_path / "mask"

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Find all instance segmentation PNG files
instance_files = sorted(instance_seg_dir.glob("instance_segmentation_*.png"))

print(f"Found {len(instance_files)} files to process")

# Process each file
for img_path in instance_files:
    # Extract file number from filename (e.g., "instance_segmentation_0018.png" -> "0018")
    match = re.search(r'instance_segmentation_(\d+)\.png', img_path.name)
    if not match:
        print(f"Skipping {img_path.name} - couldn't extract number")
        continue
    
    file_num = match.group(1)
    
    # Define corresponding files
    semantics_path = instance_seg_dir / f"instance_segmentation_semantics_mapping_{file_num}.json"
    bbox_labels_path = base_path / f"bounding_box_2d_tight/bounding_box_2d_tight_labels_{file_num}.json"
    output_path = output_dir / f"{file_num}_mask.png"
    
    # Check if required files exist
    if not semantics_path.exists():
        print(f"Skipping {file_num} - semantics mapping not found")
        continue
    if not bbox_labels_path.exists():
        print(f"Skipping {file_num} - bbox labels not found")
        continue
    
    print(f"Processing {file_num}...")
    
    # Read the RGB mask image
    rgb_mask = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if rgb_mask.shape[2] == 4:  # RGBA
        rgb_mask = rgb_mask[:, :, :3]  # Take only RGB channels

    # Read JSON files
    with open(semantics_path, 'r') as f:
        semantics_mapping = json.load(f)

    with open(bbox_labels_path, 'r') as f:
        bbox_labels = json.load(f)

    # Create class name to ID mapping from bbox_labels
    class_to_id = {}
    for id_str, info in bbox_labels.items():
        class_to_id[info['class']] = int(id_str)

    # Create single-channel semantic mask
    height, width = rgb_mask.shape[:2]
    semantic_mask = np.full((height, width), 255, dtype=np.uint8)

    # Convert RGB to class ID
    for rgba_str, info in semantics_mapping.items():
        class_name = info['class']
        
        # Skip BACKGROUND and UNLABELLED (already initialized to 255)
        if class_name in ['BACKGROUND', 'UNLABELLED']:
            continue
        
        # Parse RGBA string to RGB values
        rgba = eval(rgba_str)
        r, g, b = rgba[0], rgba[1], rgba[2]
        
        # Get class ID
        if class_name in class_to_id:
            class_id = class_to_id[class_name]
            
            # Find all pixels matching this RGB color
            mask = (rgb_mask[:, :, 2] == r) & (rgb_mask[:, :, 1] == g) & (rgb_mask[:, :, 0] == b)
            semantic_mask[mask] = class_id

    # Save the single-channel mask
    cv2.imwrite(str(output_path), semantic_mask)
    print(f"  Saved to {output_path} - Unique values: {np.unique(semantic_mask)}")

print("All files processed!")
