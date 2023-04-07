import os
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare
from ultralytics.yolo.engine.model import YOLO

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
weights = "pretrained/hair_det_512_yolov8m.onnx"


def onnx2pb(weights):
    fn, ext = os.path.splitext(os.path.normpath(weights).split(os.sep)[-1])
    weights_new = weights.replace(ext, '')

    onnx_model = onnx.load(weights.replace(ext, '.onnx'))
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(weights_new)


def pb2tflite(weights):
    fn, ext = os.path.splitext(os.path.normpath(weights).split(os.sep)[-1])

    # default
    converter = tf.lite.TFLiteConverter.from_saved_model(weights.replace(ext, ''), signature_keys=['serving_default'])
    converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_quant_model = converter.convert()

    with open(weights.replace(ext, '.tflite'), 'wb') as f_w:
        f_w.write(tflite_quant_model)

    # quantization
    converter = tf.lite.TFLiteConverter.from_saved_model(weights.replace(ext, ''), signature_keys=['serving_default'])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_quant_model = converter.convert()

    weights = weights[:-5] + '_quant.tflite'
    with open(weights, 'wb') as f_w:
        f_w.write(tflite_quant_model)


onnx_model = onnx.load(weights)
onnx2pb(weights)
pb2tflite(weights)
