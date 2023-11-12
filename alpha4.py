import cv2
import numpy as np

# 读取并显示图像
image_path = 'path_to_your_image.jpg'
image = cv2.imread(image_path)

# 显示原始图像
cv2.imshow('Original Image', image)
cv2.waitKey(0)

# 获取图像尺寸
height, width = image.shape[:2]

# 将图像转换为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 创建一个大小相同的空白图像
blank_image = np.zeros((height, width, 3), np.uint8)

# 绘制一条线段
cv2.line(blank_image, (0, 0), (width, height), (0, 255, 0), 5)

# 绘制一个矩形
cv2.rectangle(blank_image, (100, 100), (300, 300), (0, 0, 255), 3)

# 绘制一个圆
cv2.circle(blank_image, (400, 100), 50, (255, 0, 0), -1)

# 合并原始图像和绘制的图形
combined_image = cv2.addWeighted(image, 0.7, blank_image, 0.3, 0)

# 高斯模糊
blurred_image = cv2.GaussianBlur(image, (15, 15), 0)

# 边缘检测
edges = cv2.Canny(blurred_image, 30, 150)

# 腐蚀与膨胀
kernel = np.ones((5, 5), np.uint8)
eroded_image = cv2.erode(edges, kernel, iterations=1)
dilated_image = cv2.dilate(edges, kernel, iterations=1)

# 显示处理后的图像
cv2.imshow('Processed Image', combined_image)
cv2.imshow('Blurred Image', blurred_image)
cv2.imshow('Edge Detected Image', edges)
cv2.imshow('Eroded Image', eroded_image)
cv2.imshow('Dilated Image', dilated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()