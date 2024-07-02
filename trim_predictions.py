import ast
import copy
import cv2
import numpy as np

DATA_PATH = '/lulu_data/data/ssy/lululab/faceIQC/FaceDetection/LuluFaceDB_20230713_eyebrow/test'
IMG_SIZE = 96


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = copy.deepcopy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def tflite_output_to_bbox_v8(img, pred, crop_size, score_th=0.8):
    pred = pred.transpose(0, 2, 1)[0]
    img_height, img_width = img.shape[0], img.shape[1]

    obj_conf = pred[:, 4:].max(axis=1)
    conf_mask = obj_conf > score_th
    pred = pred[conf_mask]

    pred[..., 0] = pred[..., 0] / IMG_SIZE * img_width
    pred[..., 1] = pred[..., 1] / IMG_SIZE * img_height
    pred[..., 2] = pred[..., 2] / IMG_SIZE * img_width
    pred[..., 3] = pred[..., 3] / IMG_SIZE * img_height
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


def main():
    with open()

    for fn in fn_list:
        img = cv2.imread(os.path.join(DATA_PATH, fn))
        out = predict_yolo(model_det, img, crop_size=IMG_SIZE, score_th=0.8)
        write_results(img, out['obj_boxes'], fn, dst='')


if __name__ == '__main__':
    main()
