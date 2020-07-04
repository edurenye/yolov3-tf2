import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Lambda
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs


flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', './output.mp4', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


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
            max_output_size = FLAGS.yolo_max_boxes,
            iou_threshold = FLAGS.yolo_iou_threshold,
            score_threshold = FLAGS.yolo_score_threshold
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


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(FLAGS.size, classes=FLAGS.num_classes, training=True)
    else:
        yolo = YoloV3(FLAGS.size, classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    times = []

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    while True:
        _, img = vid.read()

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            break

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)
        logging.info('1 frame')

        t1 = time.time()
        #boxes, scores, classes, nums = yolo.predict(img_in)
        output_0, output_1 = yolo(img_in)
        boxes_0 = Lambda(lambda x: yolo_boxes(x, np.array([(81, 82), (135, 169),  (344, 319)], np.float32) / 416, FLAGS.num_classes),
                         name='yolo_boxes_0')(output_0)
        boxes_1 = Lambda(lambda x: yolo_boxes(x, np.array([(10, 14), (23, 27), (37, 58)], np.float32) / 416, FLAGS.num_classes),
                         name='yolo_boxes_1')(output_1)
        yolo_tiny_anchors = np.array([(10, 14), (23, 27), (37, 58), (81, 82), (135, 169), (344, 319)], np.float32) / 416
        yolo_tiny_anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])
        boxes, scores, classes, nums = Lambda(lambda x: yolo_nms(x, yolo_tiny_anchors, yolo_tiny_anchor_masks, FLAGS.num_classes),
                         name='yolo_nms')((boxes_0[:3], boxes_1[:3]))
        t2 = time.time()
        times.append(t2-t1)
        times = times[-20:]

        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        img = cv2.putText(img, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        if FLAGS.output:
            out.write(img)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
