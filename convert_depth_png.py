import numpy as np
import cv2
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from pathlib import Path

def convert_npy_to_png(input_dir, output_dir):
    """
    Convert .npy depth files to .png format
    
    Args:
        input_dir: Directory containing .npy files
        output_dir: Directory to save .png files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .npy files in the input directory
    input_path = Path(input_dir)
    npy_files = sorted(input_path.glob("distance_to_image_plane_*.npy"))
    
    print(f"Found {len(npy_files)} .npy files")
    
    for npy_file in npy_files:
        # Load depth data
        depth_data = np.load(npy_file)
        print(f"Processing {npy_file.name}, shape: {depth_data.shape}")
        print(f"  Min: {depth_data.min()}, Max: {depth_data.max()}, Mean: {depth_data.mean()}")
        frame_id = int(npy_file.stem.split('_')[-1])
        # Normalize depth to 0-65535 for 16-bit PNG
        # depth_normalized = ((depth_data - depth_data.min()) / 
        #                    (depth_data.max() - depth_data.min()) * 65535)
        depth_float32 = depth_data.astype(np.float32)
        
        # Prepare output filename
        output_filename = f"{frame_id:04d}_depth.exr"
        # output_path = output_dir / f"{frame_id:04d}_depth.exr"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save as 16-bit PNG
        cv2.imwrite(output_path, depth_float32)
        
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    input_directory = "datasets/basic1117-2/Replicator_03/distance_to_image_plane"
    output_directory = "datasets/basic1117-2/Replicator_03/depth"
    
    convert_npy_to_png(input_directory, output_directory)
    print("Conversion completed!")