import cv2
import numpy as np

path = "/media/kewei/KINGSTON/basic1117-2/Replicator_03/instance_segmentation/instance_segmentation_0006.png"

a = cv2.imread(path)
print(a.shape)

print(np.max(a))
print(np.min(a))

h, w, c = a.shape
# 将每个像素变为一行 (N, c)，然后按行去重
pixels = a.reshape(-1, c)
uniques, indices, counts = np.unique(pixels, axis=0, return_index=True, return_counts=True)
print(uniques)
print("不重复的颜色数量:", uniques.shape[0])
# print(np.array(a))

# cv2.imshow("1",a)
# cv2.waitKey(0)
