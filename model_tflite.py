from abc import ABCMeta
from tensorflow.keras.layers import Lambda

import cv2
import numpy as np
import onnx
import sys
import tensorflow as tf
import tflite_runtime.interpreter as tflite
import time

yolo_max_boxes = 100;
yolo_iou_threshold = 0.5
yolo_score_threshold = 0.5


def yolo_boxes(pred, anchors, classes):
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    grid_size = tf.shape(pred)[1]
    box_xy, box_wh, objectness, class_probs = tf.split(
        pred, (2, 2, 1, classes), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    # !!! grid[x][y] == (y, x)
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
        tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box

def yolo_loss(y_true, y_pred):
    # 1. transform all pred outputs
    # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
    pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(
        y_pred, anchors, classes)
    pred_xy = pred_xywh[..., 0:2]
    pred_wh = pred_xywh[..., 2:4]

    # 2. transform all true outputs
    # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
    true_box, true_obj, true_class_idx = tf.split(
        y_true, (4, 1, 1), axis=-1)
    true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
    true_wh = true_box[..., 2:4] - true_box[..., 0:2]

    # give higher weights to small boxes
    box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

    # 3. inverting the pred box equations
    grid_size = tf.shape(y_true)[1]
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
    true_xy = true_xy * tf.cast(grid_size, tf.float32) - \
        tf.cast(grid, tf.float32)
    true_wh = tf.math.log(true_wh / anchors)
    true_wh = tf.where(tf.math.is_inf(true_wh),
                       tf.zeros_like(true_wh), true_wh)

    # 4. calculate all masks
    obj_mask = tf.squeeze(true_obj, -1)
    # ignore false positive when iou is over threshold
    best_iou = tf.map_fn(
        lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(
            x[1], tf.cast(x[2], tf.bool))), axis=-1),
        (pred_box, true_box, obj_mask),
        tf.float32)
    ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

    # 5. calculate all losses
    xy_loss = obj_mask * box_loss_scale * \
        tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
    wh_loss = obj_mask * box_loss_scale * \
        tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
    obj_loss = binary_crossentropy(true_obj, pred_obj)
    obj_loss = obj_mask * obj_loss + \
        (1 - obj_mask) * ignore_mask * obj_loss
    # TODO: use binary_crossentropy instead
    class_loss = obj_mask * sparse_categorical_crossentropy(
        true_class_idx, pred_class)

    # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
    xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
    wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
    obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
    class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

    return xy_loss + wh_loss + obj_loss + class_loss

def yolo_nms(outputs, anchors, masks, classes):
    # boxes, conf, type
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    scores = confidence * class_probs

    all_boxes = tf.reshape(bbox, (-1, 4))
    all_scores = tf.reshape(scores, (-1, tf.shape(scores)[-1]))
    num_classes = classes
    my_structure = [{'boxes': [], 'scores': [], 'classes': []}] * num_classes
    all_classes = tf.argmax(all_scores, axis=1)
    all_scores = tf.reduce_max(all_scores, axis=1)
    for i in range(len(all_classes)):
        c = all_classes[i]
        my_structure[c]['boxes'].append(all_boxes[i])
        my_structure[c]['scores'].append(all_scores[i])
        my_structure[c]['classes'].append(tf.cast(c, tf.float32))
    all_nms_index = []
    boxes = [[]]
    scores = [[]]
    classes = [[]]
    for c in range(num_classes):
        nms_index = tf.image.non_max_suppression(
            boxes = my_structure[c]['boxes'],
            scores = my_structure[c]['scores'],
            max_output_size = yolo_max_boxes,
            iou_threshold = yolo_iou_threshold,
            score_threshold = yolo_score_threshold
        )
        all_nms_index.extend(nms_index)
        boxes[0].extend(tf.gather(my_structure[c]['boxes'], nms_index))
        scores[0].extend(tf.gather(my_structure[c]['scores'], nms_index))
        classes[0].extend(tf.gather(my_structure[c]['classes'], nms_index))
    boxes = tf.convert_to_tensor(boxes, dtype=tf.float32)
    scores = tf.convert_to_tensor(scores, dtype=tf.float32)
    classes = tf.convert_to_tensor(classes, dtype=tf.float32)
    valid_detections = tf.convert_to_tensor([tf.size(all_nms_index)], dtype=tf.int32)

    return boxes, scores, classes, valid_detections

def transform_images(img, size):
    img = tf.image.resize(img, (size, size))
    img = img / 255
    return img

def draw_outputs(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, '{} {:.4f}'.format(
            class_names[int(classes[i])], objectness[i]),
            x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img

def load_model(app, model_type, model_path, classes_path, size = 416):
    return PigallModel.load_model(app, model_type, model_path, classes_path, size)


class PigallModel(metaclass=ABCMeta):

    def __init__(self, app, model_type, model_path, classes_path, size):
        self.app = app
        self.size = size
        self.model_type = model_type
        self.class_names = self.get_class_names(classes_path)

    @staticmethod
    def load_model(app, model_type, model_path, classes_path, size):
        if model_type == 'onnx':
            return OnnxPigallModel(app, model_type, model_path, classes_path, size)
        elif model_type == 'cv-tf':
            return CVTFPigallModel(app, model_type, model_path, classes_path, size)
        elif model_type == 'tf':
            return TFPigallModel(app, model_type, model_path, classes_path, size)
        elif model_type == 'tflite':
            return TFLitePigallModel(app, model_type, model_path, classes_path, size)
        else:
            return None

    def get_class_names(self, classes_path):
        class_names = [c.strip() for c in open(classes_path).readlines()]
        print('We got ' + str(len(class_names)) + ' classes.', file=sys.stderr)
        return class_names

    def make_prediction(self, inputs):
        img = tf.expand_dims(inputs, 0)
        img = transform_images(img, self.size)

        t1 = time.time()
        output_0, output_1 = self.model.predict(img)
        t2 = time.time()
        prediction_time = t2 - t1

        boxes, scores, classes, nums = self.apply_nms(output_0, output_1)
        img = cv2.cvtColor(inputs, cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), self.class_names)
        img = cv2.putText(img, "Time: {:.2f}ms".format(prediction_time*1000), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        return img

    def detect(self, currentFrame):
        while True:
            img = currentFrame
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = tf.expand_dims(img, 0)
                img = transform_images(img, self.size)

                t1 = time.time()
                boxes, scores, classes, nums = self.make_prediction(img)
                t2 = time.time()
                times.append(t2-t1)
                times = times[-20:]
                print('Prediction done in ' + times + ' seconds.', file=sys.stderr)

                img = draw_outputs(img, (boxes, scores, classes, nums), self.class_names)
                img = cv2.putText(img, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

                return img
            else:
                return None

    def apply_nms(self, output_0, output_1):
        boxes_0 = Lambda(lambda x: yolo_boxes(x, np.array([(81, 82), (135, 169),  (344, 319)], np.float32) / 416, len(self.class_names)),
                         name='yolo_boxes_0')(output_0)
        boxes_1 = Lambda(lambda x: yolo_boxes(x, np.array([(10, 14), (23, 27), (37, 58)], np.float32) / 416, len(self.class_names)),
                         name='yolo_boxes_1')(output_1)
        yolo_tiny_anchors = np.array([(10, 14), (23, 27), (37, 58), (81, 82), (135, 169), (344, 319)], np.float32) / 416
        yolo_tiny_anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])
        boxes, scores, classes, nums = Lambda(lambda x: yolo_nms(x, yolo_tiny_anchors, yolo_tiny_anchor_masks, len(self.class_names)),
                                              name='yolo_nms')((boxes_0[:3], boxes_1[:3]))
        return boxes, scores, classes, nums


class OnnxPigallModel(PigallModel):

    def __init__(self, app, model_type, model_path, classes_path, size):
        super().__init__(app, model_type, model_path, classes_path, size)
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        self.model = cv2.dnn.readNetFromONNX(model_path)
        print('Model loaded.', file=sys.stderr)


class CVTFPigallModel(PigallModel):

    def __init__(self, app, model_type, model_path, classes_path, size):
        super().__init__(app, model_type, model_path, classes_path, size)
        self.model = cv2.dnn.readNetFromTensorflow(model_path, './pbtxt/network.pbtxt')
        print('Model loaded.', file=sys.stderr)


class TFPigallModel(PigallModel):

    def __init__(self, app, model_type, model_path, classes_path, size):
        super().__init__(app, model_type, model_path, classes_path, size)
        self.model = tf.keras.models.load_model(model_path, custom_objects={'tf': tf, 'yolo_boxes': yolo_boxes, 'yolo_loss': yolo_loss})
        print('Model loaded.', file=sys.stderr)

    def make_prediction(self, inputs):
        outputs = self.model.predict(inputs)
        return self.model.predict(inputs)


class TFLitePigallModel(PigallModel):

    def __init__(self, app, model_type, model_path, classes_path, size):
        super().__init__(app, model_type, model_path, classes_path, size)
        self.interpreter = tflite.Interpreter(model_path = model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print('Model loaded.', file=sys.stderr)

    def make_prediction(self, inputs):
        img = tf.expand_dims(inputs, 0)
        img = transform_images(img, self.size)
        self.interpreter.set_tensor(self.input_details[0]['index'], img)

        t1 = time.time()
        self.interpreter.invoke()
        t2 = time.time()
        prediction_time = t2 - t1

        output_0 = self.interpreter.get_tensor(self.output_details[0]['index'])
        output_1 = self.interpreter.get_tensor(self.output_details[1]['index'])
        self.interpreter.reset_all_variables()
        boxes, scores, classes, nums = self.apply_nms(output_0, output_1)
        img = cv2.cvtColor(inputs, cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), self.class_names)
        img = cv2.putText(img, "Time: {:.2f}ms".format(prediction_time*1000), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        return img
