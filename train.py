#coding:utf-8
from ultralytics import YOLO

# 加载yolov8预训练模型
model = YOLO("yolov8n-seg.pt")
# Use the model
if __name__ == '__main__':
    # Use the model
    results = model.train(data='datasets/Data/data.yaml', epochs=250, batch=4)  # 训练模型
    # 将模型转为onnx格式
    # success = model.export(format='onnx')



