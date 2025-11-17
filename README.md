# Isaac Sim 数据可视化工具

这个工具可以读取Isaac Sim生成的数据并在RViz中可视化点云和3D边界框。

## 1. 相机参数转换

### 1.1 输入参数（来自Isaac Sim）

从Isaac Sim的相机参数JSON文件中读取：

| 参数名 | 符号 | 说明 | 单位 |
|--------|------|------|------|
| `cameraAperture` | $(W_{sensor}, H_{sensor})$ | 传感器物理尺寸 | mm |
| `cameraFocalLength` | $f_{mm}$ | 焦距 | mm |
| `renderProductResolution` | $(W_{img}, H_{img})$ | 图像分辨率 | pixels |
| `cameraApertureOffset` | $(O_x, O_y)$ | 传感器偏移 | mm |

### 1.2 像素焦距转换公式

$$
f_x = f_{mm} \times \frac{W_{img}}{W_{sensor}}
$$

$$
f_y = f_{mm} \times \frac{H_{img}}{H_{sensor}}
$$

### 1.3 主点坐标转换公式

$$
c_x = \frac{W_{img}}{2} + O_x \times \frac{W_{img}}{W_{sensor}}
$$

$$
c_y = \frac{H_{img}}{2} + O_y \times \frac{H_{img}}{H_{sensor}}
$$

### 1.4 相机内参矩阵

$$
K = \begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}
$$

### 1.5 深度图到点云转换

对于图像坐标 $(u, v)$ 和深度值 $d$，转换到相机坐标系下的3D点 $(x, y, z)$：

$$
\begin{cases}
x = \frac{(u - c_x) \times d}{f_x} \\
y = \frac{(v - c_y) \times d}{f_y} \\
z = d
\end{cases}
$$

齐次坐标形式：

$$
\begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix} = 
d \cdot K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
$$

## 2. 3D边界框坐标变换

### 2.1 坐标系定义

- **局部坐标系** $(P_{local})$: 物体边界框的局部坐标系
- **世界坐标系** $(P_{world})$: Isaac Sim的世界坐标系
- **相机坐标系** $(P_{camera})$: Isaac Sim相机坐标系
- **ROS相机坐标系** $(P_{ros})$: ROS2标准相机坐标系

### 2.2 变换矩阵定义

#### 2.2.1 局部到世界变换矩阵

每个边界框包含一个4×4变换矩阵 $T_{local \rightarrow world}$：

$$
T_{local \rightarrow world} = \begin{bmatrix}
R_{11} & R_{12} & R_{13} & t_x \\
R_{21} & R_{22} & R_{23} & t_y \\
R_{31} & R_{32} & R_{33} & t_z \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

其中：
- $R_{3\times3}$ 是旋转矩阵
- $\mathbf{t} = [t_x, t_y, t_z]^T$ 是平移向量

#### 2.2.2 世界到相机变换矩阵

从 `cameraViewTransform` 获取并转置（行主序→列主序）：

$$
T_{world \rightarrow camera} = (T_{view})^T
$$

#### 2.2.3 相机坐标系转换矩阵

绕X轴旋转180°，将Isaac Sim相机坐标系转换为ROS标准：

$$
T_{camera \rightarrow ros} = R_x(180°) = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & -1 & 0 & 0 \\
0 & 0 & -1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

作用：
- X轴不变（右方向）
- Y轴翻转：上→下
- Z轴翻转：向后→向前

### 2.3 完整变换公式

#### 2.3.1 组合变换矩阵

$$
T_{complete} = T_{camera \rightarrow ros} \cdot T_{world \rightarrow camera} \cdot T_{local \rightarrow world}
$$

展开形式：

$$
T_{complete} = R_x(180°) \cdot (T_{view})^T \cdot T_{bbox}
$$

#### 2.3.2 边界框顶点变换

对于边界框的8个顶点 $\mathbf{p}_i^{local}$：

$$
\mathbf{p}_i^{ros} = T_{complete} \cdot \tilde{\mathbf{p}}_i^{local}
$$

其中 $\tilde{\mathbf{p}}_i^{local}$ 是齐次坐标：

$$
\tilde{\mathbf{p}}_i^{local} = \begin{bmatrix} x_i \\ y_i \\ z_i \\ 1 \end{bmatrix}
$$

#### 2.3.3 边界框8个顶点定义

在局部坐标系下：

$$
\begin{aligned}
\mathbf{p}_0 &= [x_{min}, y_{min}, z_{min}]^T \quad \text{(底-前-左)} \\
\mathbf{p}_1 &= [x_{max}, y_{min}, z_{min}]^T \quad \text{(底-前-右)} \\
\mathbf{p}_2 &= [x_{max}, y_{max}, z_{min}]^T \quad \text{(底-后-右)} \\
\mathbf{p}_3 &= [x_{min}, y_{max}, z_{min}]^T \quad \text{(底-后-左)} \\
\mathbf{p}_4 &= [x_{min}, y_{min}, z_{max}]^T \quad \text{(顶-前-左)} \\
\mathbf{p}_5 &= [x_{max}, y_{min}, z_{max}]^T \quad \text{(顶-前-右)} \\
\mathbf{p}_6 &= [x_{max}, y_{max}, z_{max}]^T \quad \text{(顶-后-右)} \\
\mathbf{p}_7 &= [x_{min}, y_{max}, z_{max}]^T \quad \text{(顶-后-左)}
\end{aligned}
$$

### 2.4 矩阵运算实现

```python
# 组合变换矩阵
combined_transform = rot_x_180 @ cam_view_matrix @ transform

# 局部顶点（齐次坐标）
corners_local_homogeneous = np.hstack([corners_local, np.ones((8, 1))])

# 应用变换
corners_ros_homogeneous = (combined_transform @ corners_local_homogeneous.T).T

# 提取3D坐标
corners_ros = corners_ros_homogeneous[:, :3]
```

## 3. 坐标系约定

### 3.1 ROS相机坐标系 (`camera_frame`)

$$
\begin{aligned}
\text{X轴} &: \text{向右} \\
\text{Y轴} &: \text{向下} \\
\text{Z轴} &: \text{向前（场景深度方向）}
\end{aligned}
$$

符合OpenCV/ROS标准相机坐标系约定。

## 4. 使用方法

### 4.1 启动可视化

```bash
# 终端1 - 启动ROS2节点
python3 vis_simdata.py

# 终端2 - 启动RViz
rviz2
```

### 4.2 RViz配置

- **Fixed Frame**: `camera_frame`
- **PointCloud2** 话题: `/pointcloud`
- **MarkerArray** 话题: `/bounding_boxes`

## 5. 数据格式

| 文件类型 | 文件名模式 | 说明 |
|---------|-----------|------|
| RGB图像 | `rgb_{id:04d}.png` | 彩色图像 |
| 深度图 | `distance_to_camera_{id:04d}.npy` | 深度数据 |
| 2D边界框 | `bounding_box_2d_tight_{id:04d}.npy` | 2D紧凑边界框 |
| 3D边界框 | `bounding_box_3d_{id:04d}.npy` | 3D边界框（包含变换矩阵） |
| 相机参数 | `camera_params_{id:04d}.json` | 相机内外参数 |
| 标签 | `*_labels_{id:04d}.json` | 类别标签映射 |

## 6. 边界框可视化

### 6.1 12条边的定义

- **底面**: $(p_0, p_1), (p_1, p_2), (p_2, p_3), (p_3, p_0)$
- **顶面**: $(p_4, p_5), (p_5, p_6), (p_6, p_7), (p_7, p_4)$
- **竖边**: $(p_0, p_4), (p_1, p_5), (p_2, p_6), (p_3, p_7)$

### 6.2 颜色编码

| 类别 | RGB | 颜色 |
|------|-----|------|
| `nvidia_cube` | $(1.0, 0.0, 0.0)$ | 红色 |
| `chess_box` | $(0.0, 1.0, 0.0)$ | 绿色 |
| `mug` | $(0.0, 0.0, 1.0)$ | 蓝色 |

## 7. ROS2话题

| 话题名 | 消息类型 | 频率 | 说明 |
|--------|---------|------|------|
| `/pointcloud` | `sensor_msgs/PointCloud2` | 1 Hz | 彩色点云 |
| `/bounding_boxes` | `visualization_msgs/MarkerArray` | 1 Hz | 3D边界框 |
