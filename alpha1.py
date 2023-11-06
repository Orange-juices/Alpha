import cv2
import numpy as np

# 加载模型和类别标签
model = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# 设置输入尺寸和阈值
input_size = 416
conf_threshold = 0.5
nms_threshold = 0.4

# 加载图像并进行预处理
img = cv2.imread("test.jpg")
blob = cv2.dnn.blobFromImage(img, 1/255, (input_size, input_size), swapRB=True, crop=False)

# 将图像置入模型中进行前向推理
model.setInput(blob)
output_layers_names = model.getUnconnectedOutLayersNames()
layerOutputs = model.forward(output_layers_names)

# 解析输出结果并筛选符合条件的检测框
boxes = []
confidences = []
classIDs = []

for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > conf_threshold:
            center_x = int(detection[0] * img.shape[1])
            center_y = int(detection[1] * img.shape[0])
            w = int(detection[2] * img.shape[1])
            h = int(detection[3] * img.shape[0])

            x = int(center_x - w/2)
            y = int(center_y - h/2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            classIDs.append(classID)

# 极大值抑制
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# 标注检测结果
if len(indices) > 0:
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = classes[classIDs[i]]
        confidence = confidences[i]
        color = (0, 255, 0)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        text = "{}: {:.4f}".format(label, confidence)
        cv2.putText(img, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# 显示并保存结果
cv2.imshow("YOLOv3", img)
cv2.imwrite("result.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
