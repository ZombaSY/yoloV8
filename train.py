from ultralytics.yolo.engine.model import YOLO
import os


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Load a model
weights = "pretrained/hair_det_512_yolov8s.pt"
model = YOLO(weights)  # load a pretrained model (recommended for training)

# Use the model
# model.train(data="pretrained/hair_detection.yaml", epochs=1000, imgsz=512, batch=64)  # train the model
model.val(data="pretrained/hair_detection.yaml")
# success = model.export(format="onnx", imgsz=512)  # export the model to ONNX format
