from absl import app, flags, logging
from absl.flags import FLAGS

from model_tflite import load_model
import cv2

flags.DEFINE_string('classes', './data/_classes.txt', 'path to classes file')
#flags.DEFINE_string('model', './model/keras_yolov3-tiny.tflite', 'path to model file')
flags.DEFINE_string('model', './model/model.onnx', 'path to model file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', '', 'path to input image')
flags.DEFINE_string('video', '', 'path to input image')
flags.DEFINE_string('output', '', 'path to output image')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')


def main(_argv):
    #model = load_model(app, model_type = 'tflite', model_path = FLAGS.model, classes_path = FLAGS.classes)
    model = load_model(app, model_type = 'onnx', model_path = FLAGS.model, classes_path = FLAGS.classes)
    logging.info('OpenCV version: ' + cv2.__version__)

    if FLAGS.image:
        img_raw = cv2.imread(FLAGS.image)
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

        img = model.make_prediction(img_raw)

        cv2.imwrite(FLAGS.output, img)
        logging.info('output saved to: {}'.format(FLAGS.output))
    
    if FLAGS.video:
        try:
            vid = cv2.VideoCapture(int(FLAGS.video))
        except:
            vid = cv2.VideoCapture(FLAGS.video)

        image_size = FLAGS.size
        width  = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        offsetW = int(round((width - image_size) / 2))
        offsetH = int(round((height - image_size) / 2))
        
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
        
        while True:
            _, img_raw = vid.read()

            if img_raw is None:
                logging.warning("Empty Frame")
                break
            logging.warning("Got one frame")
            
            img_raw = img_raw[offsetH:offsetH+image_size, offsetW:offsetW+image_size]
            cv2.imwrite('./debug.jpg', img_raw)

            img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

            img = model.make_prediction(img_raw)
            cv2.imwrite('./debug2.jpg', img)

            out.write(img)
        logging.info('output saved to: {}'.format(FLAGS.output))
        out.release()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
