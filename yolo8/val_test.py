from ultralytics import YOLO

# 加载已保存的模型权重 (将路径替换为你实际保存模型的路径)
model = YOLO('runs/detect/train/weights/best.pt')
