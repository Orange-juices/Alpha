from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# 初始化YOLO模型
yolo = YOLO('weights/best.pt')

# 图像路径
image_path = "000025.jpg"

# 执行实例分割
results = yolo.predict("000025.jpg")
# print(results[0].masks)

# 获取分割结果
pixel_xy = results[0].masks.xy[0]
points = np.array(pixel_xy, np.int32)


original_image = cv2.imread(image_path)
mask_image = np.zeros_like(original_image)

# 将轮廓绘制到掩码图像上
cv2.drawContours(mask_image, [points], -1, (255, 255, 255), thickness=cv2.FILLED)

# 使用掩码选择轮廓内的像素
masked_pixels = cv2.bitwise_and(original_image, mask_image)

# 计算轮廓内像素的RGB均值
mean_rgb = np.mean(masked_pixels, axis=(0, 1))

# 分别输出RGB通道的均值
mean_r, mean_g, mean_b = mean_rgb

# 输出RGB通道的均值
print("Mean R:", mean_r)
print("Mean G:", mean_g)
print("Mean B:", mean_b)


#计算HIS颜色空间
hsv_image = cv2.cvtColor(masked_pixels, cv2.COLOR_BGR2HSV)
# 计算轮廓内像素的HIS通道均值
mean_hsv = np.mean(hsv_image, axis=(0, 1))

# 分别输出HIS通道的均值
mean_h, mean_s, mean_v = mean_hsv

# 输出HIS通道的均值
print("Mean H:", mean_h)
print("Mean S:", mean_s)
print("Mean V:", mean_v)