import cv2
import json
import numpy as np
from pathlib import Path

# Define paths
base_path = Path("datasets/basic1117-2/Replicator_03")
img_path = base_path / "instance_segmentation/instance_segmentation_0018.png"
semantics_path = base_path / "instance_segmentation/instance_segmentation_semantics_mapping_0018.json"
bbox_labels_path = base_path / "bounding_box_2d_tight/bounding_box_2d_tight_labels_0018.json"
output_path = Path("0018_mask.png")

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
print(f"Semantic mask saved to {output_path}")
print(f"Unique values in mask: {np.unique(semantic_mask)}")
