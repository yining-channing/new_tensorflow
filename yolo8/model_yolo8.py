from ultralytics import YOLO

# 1. 加载预训练模型
model = YOLO('yolov8n.pt')  # 使用轻量级 YOLOv8n 模型

# 2. 开始训练
model.train(data=r'E:\gitrepo\yolo8\dataset.yaml', epochs=5, batch=16, imgsz=640)

# 3. 评估训练好的模型
metrics = model.val(data=r'E:\gitrepo\yolo8\dataset.yaml')

# 4. 测试某张图片
results = model('E:/shuju/video_frames/images/test/frame_10400.jpg')
results.show()

# 5. 保存模型
model.save('best_model.pt')
