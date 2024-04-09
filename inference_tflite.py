import cv2
import numpy as np
import copy
import os
import tensorflow as tf
import sys


DATA_PATH = '/lulu_data/data/ssy/lululab/faceIQC/FaceDetection/LuluFaceDB_20230713_eyebrow/test'
MODEL_PATH = 'pretrained/best-float32-NCHW.tflite'

sys.path.append(os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def write_results(img, obj_boxes, fn, dst=None):
    if obj_boxes is not None:
        for i in range(len(obj_boxes)):
            p1 = [int(obj_boxes[i][0]), int(obj_boxes[i][1])]
            p2 = [int(obj_boxes[i][2]), int(obj_boxes[i][3])]

            cv2.rectangle(img, p1, p2, (0, 0, 255), 2)

    dst = os.path.join('results', dst)
    if not os.path.exists(dst):
        os.makedirs(dst)

    cv2.imwrite(os.path.join(dst, fn), img.astype(np.uint8))


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = copy.deepcopy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def tflite_output_to_bbox_v8(img, pred, crop_size, score_th):
    pred = pred.transpose(0, 2, 1)[0]
    img_height, img_width = img.shape[0], img.shape[1]

    obj_conf = pred[:, 4:].max(axis=1)
    conf_mask = obj_conf > score_th
    pred = pred[conf_mask]

    pred[..., 0] = pred[..., 0] / 128 * img_width
    pred[..., 1] = pred[..., 1] / 128 * img_height
    pred[..., 2] = pred[..., 2] / 128 * img_width
    pred[..., 3] = pred[..., 3] / 128 * img_height
    obj_conf = obj_conf[conf_mask]
    obj_classes = pred[:, 4:].argmax(axis=1)
    obj_boxes_xywh = pred[:, :4]
    obj_boxes_xyxy = xywh2xyxy(pred[:, :4])

    # NMS
    out = cv2.dnn.NMSBoxes(obj_boxes_xyxy, obj_conf, score_th, 0.50)
    obj_boxes = obj_boxes_xyxy[out]
    obj_classes = obj_classes[out]
    obj_boxes_xywh = obj_boxes_xywh[out]
    obj_boxes = np.concatenate([obj_boxes, np.expand_dims(obj_classes, axis=1)], axis=1)

    img_list = []
    mask_list = []

    for i in range(len(obj_boxes)):
        img_info = {}
        crop_size_half = crop_size // 2

        cx, cy = [int(obj_boxes_xywh[i][0]), int(obj_boxes_xywh[i][1])]
        x1 = cx - crop_size_half if cx - crop_size_half >= 0 else 0
        x2 = cx + crop_size_half if cx + crop_size_half < img_width else img_width - 1
        y1 = cy - crop_size_half if cy - crop_size_half >= 0 else 0
        y2 = cy + crop_size_half if cy + crop_size_half < img_height else img_height - 1

        img_crop = img[y1:y2, x1:x2]
        mask_list.append(i)
        img_info['image'] = cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR)
        img_list.append(img_info)

    obj_boxes = obj_boxes[mask_list]
    out_dict = {'image': np.stack(item['image'] for item in img_list) if len(img_list) != 0 else None,
                'hairCount': sum(obj_boxes[..., 4] + 1) if len(img_list) != 0 else None,
                'obj_boxes': obj_boxes if len(img_list) != 0 else None}
    return out_dict


def predict_yolo(model, img, crop_size, score_th):
    img_rsz = cv2.resize(img, (128, 128))
    img_rsz = cv2.cvtColor(img_rsz, cv2.COLOR_BGR2RGB)
    img_tf = img_rsz / 255.0
    img_tf = img_tf.astype(np.float32)

    img_tf = np.transpose(img_tf, [2, 0, 1])
    img_tf = img_tf.astype(np.float32)
    img_tf = np.expand_dims(img_tf, axis=0)
    img_tf = tf.convert_to_tensor(img_tf)

    input_details = model.get_input_details()
    output_details = model.get_output_details()
    model.set_tensor(input_details[0]['index'], img_tf)
    model.invoke()
    output_data = model.get_tensor(output_details[0]['index'])

    out_dict = tflite_output_to_bbox_v8(img, output_data, crop_size, score_th=score_th)

    return out_dict


def main():
    fn_list = os.listdir(DATA_PATH)
    model_det = tf.lite.Interpreter(MODEL_PATH)
    model_det.allocate_tensors()

    for fn in fn_list:
        img = cv2.imread(os.path.join(DATA_PATH, fn))
        out = predict_yolo(model_det, img, crop_size=128, score_th=0.8)
        write_results(img, out['obj_boxes'], fn, dst='')


if __name__ == '__main__':
    main()
