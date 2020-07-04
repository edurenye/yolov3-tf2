from absl import app, flags, logging

import cv2
import numpy as np
import tensorflow as tf
import time


def load_model():
    return tf.keras.models.load_model('yolov3-tiny.h5', custom_objects={'tf': tf, 'yolo_boxes': yolo_boxes})

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

def transform_images(img, size):
    img = tf.image.resize(img, size)
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

def main(_argv):
    model = load_model()
    img = tf.image.decode_image(open('data/street.jpg', 'rb').read(), channels=3)
    img = tf.expand_dims(img, 0)
    img = transform_images(img, size)

    t1 = time.time()
    boxes, scores, classes, nums = model.predict(img)
    t2 = time.time()
    times.append(t2-t1)
    times = times[-20:]

    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    img = cv2.putText(img, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

    cv2.imwrite('data/street_out_pigall.jpg', img)
    logging.info('output saved to: {}'.format('data/street_out_pigall.jpg'))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
