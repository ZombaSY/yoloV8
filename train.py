from ultralytics.yolo.engine.model import YOLO
import os
import sys

sys.path.append(os.getcwd())

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Load a model
weights = 'pretrained/best.pt'   # runs/detect/train4/weights/best.pt, pretrained/yolov8m.pt
model = YOLO(weights)  # load a pretrained model (recommended for training)

# hair_detection
# model.train(data='pretrained/hair_detection.yaml', epochs=1000, imgsz=512, batch=64, mixup=0.2, hsv_h=0.5, hsv_s=0.5, hsv_v=0.5, patience=100, lr0=5e-6, optimizer='AdamW', cos_lr=True, warmup_epochs=10)  # train the model
# model.val(data="pretrained/hair_detection.yaml")
# success = model.export(format='tflite', imgsz=512)  # export the model

# face_detection
model.train(data='pretrained/face_detection.yaml', epochs=1000, imgsz=128, batch=1024, hsv_h=0.5, hsv_s=0.5, hsv_v=0.5, patience=30, lr0=5e-6, optimizer='AdamW', cos_lr=True, warmup_epochs=10)  # train the model
# model.val(data="pretrained/face_detection.yaml", imgsz=128, batch=256)
# success = model.export(format='tflite', imgsz=128, int8=True)  # export the model
