from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import YoloV3, YoloV3Tiny
from yolov3_tf2.utils import load_darknet_weights
from tensorflow.python.client import session
from tensorflow.python.framework import graph_util

flags.DEFINE_string('weights', './data/yolov3.weights', 'path to weights file')
flags.DEFINE_string('output', 'yolov3_graph.pb', 'path to output')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_integer('size', 416, 'image size')

def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(size=FLAGS.size, classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(size=FLAGS.size, classes=FLAGS.num_classes)
    yolo.summary()
    #session = session.Session()
    #tf.train.write_graph(session.graph_def, "./export", "network.pb", False)
    logging.info('model exported')

    load_darknet_weights(yolo, FLAGS.weights, FLAGS.tiny)
    logging.info('weights loaded')

    img = np.random.random((1, FLAGS.size, FLAGS.size, 3)).astype(np.float32)
    output = yolo(img)
    logging.info('sanity check passed')

    with session.Session() as sess:
        frozen_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['output'])
        tf.io.write_graph(frozen_graph, './logs', './data/frozeen_graph/' + FLAGS.output, as_text=False)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass