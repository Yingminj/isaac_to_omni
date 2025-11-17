# Isaac Sim 数据可视化工具

这个工具可以读取Isaac Sim生成的数据并在RViz中可视化点云和3D边界框。

## 功能

1. 从深度图和RGB图生成彩色点云
2. 在点云中显示3D边界框
3. 为不同类别的物体显示不同颜色的边界框
4. 显示物体类别标签

## 依赖项

```bash
# ROS2依赖
sudo apt install ros-humble-rviz2 ros-humble-visualization-msgs

# Python依赖
pip install opencv-python numpy
```

## 使用方法

### 方法1：使用启动脚本

```bash
chmod +x launch_visualizer.sh
./launch_visualizer.sh
```

### 方法2：手动启动

终端1 - 启动ROS2节点：
```bash
python3 simdata_to_omni.py
```

终端2 - 启动RViz：
```bash
rviz2 -d visualize.rviz
```

## 数据格式

脚本会读取以下文件：
- `test/rgb_0000.png` - RGB图像
- `test/distance_to_camera_0000.npy` - 深度图
- `test/instance_segmentation_0000.png` - 实例分割图
- `test/instance_segmentation_mapping_0000.json` - 实例映射
- `test/instance_segmentation_semantics_mapping_0000.json` - 语义映射
- `test/bounding_box_2d_tight_0000.npy` - 2D边界框
- `test/bounding_box_2d_tight_labels_0000.json` - 2D标签
- `test/bounding_box_2d_tight_prim_paths_0000.json` - 2D路径
- `test/bounding_box_3d_0000.npy` - 3D边界框
- `test/bounding_box_3d_labels_0000.json` - 3D标签
- `test/bounding_box_3d_prim_paths_0000.json` - 3D路径

## ROS2话题

- `/pointcloud` - 彩色点云数据 (sensor_msgs/PointCloud2)
- `/bounding_boxes` - 3D边界框标记 (visualization_msgs/MarkerArray)

## 边界框颜色

- nvidia_cube: 红色
- chess_box: 绿色
- mug: 蓝色

## 坐标系

- Fixed Frame: `camera_frame`

## 调整参数

如需调整相机内参，请编辑 `simdata_to_omni.py` 中的 `depth_to_pointcloud` 方法：

```python
def depth_to_pointcloud(self, rgb, depth, fx=525.0, fy=525.0, cx=None, cy=None):
```

修改 `fx` 和 `fy` 参数以匹配您的相机焦距。
