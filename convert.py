from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
from yolov3_tf2.models import YoloV3, YoloV3Tiny
from yolov3_tf2.utils import load_darknet_weights
import tensorflow as tf

flags.DEFINE_string('weights', './data/yolov3.weights', 'path to weights file')
flags.DEFINE_string('output', './checkpoints/yolov3', 'path to output')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_boolean('save_model', False, 'save model or just the weights')
flags.DEFINE_integer('size', 416, 'image size')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(size=FLAGS.size, classes=FLAGS.num_classes, training=True)
    else:
        yolo = YoloV3(size=FLAGS.size, classes=FLAGS.num_classes)
    yolo.summary()
    logging.info('model created')

    yolo.load_weights(FLAGS.weights).expect_partial()
    #load_darknet_weights(yolo, FLAGS.weights, FLAGS.tiny)
    logging.info('weights loaded')

    img = np.random.random((1, 416, 416, 3)).astype(np.float32)
    output = yolo(img)
    logging.info('sanity check passed')

    if not FLAGS.save_model:
        yolo.save_weights(FLAGS.output + '.tf')
        logging.info('weights saved')
    else:
        yolo.save('./model/keras_' + FLAGS.output)
        logging.info('keras model saved')
        
        converter = tf.lite.TFLiteConverter.from_keras_model(yolo)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        open('./model/keras_' + FLAGS.output + '.tflite', 'wb').write(tflite_model)
        logging.info('keras lite model saved')
        
        export_dir = './model/tf_' + FLAGS.output
        tf.saved_model.save(yolo, export_dir)
        logging.info('tf model saved')
        model = tf.saved_model.load(export_dir)
        concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        concrete_func.inputs[0].set_shape([1, 416, 416, 3])
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()
        open('./model/' + FLAGS.output + '.tflite', 'wb').write(tflite_model)
        logging.info('lite model saved')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
