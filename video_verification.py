from yolov3_tf2.utils import draw_outputs
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from absl.flags import FLAGS
from absl import app, flags, logging
import time
import os
import glob

import cv2
import matplotlib.pyplot as plt
import dlib

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Softmax, Flatten, Activation, BatchNormalization
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow.keras.backend as K

from scipy.spatial.distance import cosine


import random
import websockets
import asyncio
import json
import base64


def get_match(known_embeddings, candidate_embedding, thresh=0.3):

    minim = thresh
    person = "unknown"

    for key in known_embeddings:
        score = cosine(known_embeddings[key], candidate_embedding)
        if minim > score:
            minim = score
            person = key

    return person, minim


def recognize(img, outputs, class_names, vgg_face):
    person_rep = dict()
    person_names = ["angelamerkel", "jinping", "trump"]
    for person in person_names:
        embed = np.loadtxt(person+".txt")
        person_rep[person] = embed

    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        if class_names[int(classes[i])] == "face":

            img_crop = img[x1y1[1]:x2y2[1], x1y1[0]:x2y2[0]]

            crop_img = img_to_array(img_crop)
            crop_img = np.expand_dims(crop_img, axis=0)
            crop_img = preprocess_input(crop_img)

            img_encode = vgg_face(transform_images(crop_img, 224))
            embed = K.eval(img_encode)
            name, score = get_match(person_rep, embed, 0.3)

            img = cv2.rectangle(img, x1y1, x2y2, (205, 0, 0), 2)
            img = cv2.putText(img, '{} {:.4f}'.format(
                name, score),
                x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
        else:
            img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
            img = cv2.putText(img, '{} {:.4f}'.format(
                class_names[int(classes[i])], objectness[i]),
                x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img


flags.DEFINE_string('classes', './data/face.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3_train_10.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './DonaldTrump.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID',
                    'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


class VideoCamera(object):
    def __init__(self, stream):
        self.video = stream

    def __del__(self):
        self.video

    def get_frame(self):
        while True:
            image = self.video
            ret, jpeg = cv2.imencode('.jpg', image)
            base6 = base64.b64encode(jpeg.tobytes())
            yield base6.decode('utf-8')


def gen(camera):

    while True:
        image = next(camera.get_frame())

        yield(image)


def recog(_argv):

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')
    vgg_face = tf.keras.models.load_model("vgg_model.h5")
    logging.info('vgg_model loaded')
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
            continue

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        t2 = time.time()
        times.append(t2-t1)
        times = times[-20:]

        img = recognize(img, (boxes, scores, classes, nums),
                        class_names, vgg_face)
        img = cv2.putText(img, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        if FLAGS.output:
            out.write(img)
        yield img
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()


async def sender(websocket, path):
    camera = VideoCamera(recog)
    i = 0
    while True:
        i = i+1
        data = next(gen(camera))
        js = {'data': data, "infos": [
            {
                "key": "1",
                "name": "John Brown",
                "age": 32,
                "address": "New York No. 1 Lake Park",
            },
            {
                "key": "2",
                "name": "Jim Green",
                "age": 42,
                "address": "London No. 1 Lake Park",
            },
            {
                "key": "3",
                "name": "Joe Black",
                "age": 32,
                "address": "Sidney No. 1 Lake Park",
            },
            {
                "key": "4",
                "name": "Jim Red",
                "age": 32,
                "address": "London No. 2 Lake Park",
            },
        ], }

        res_bytes = json.dumps(js).encode('utf-8')
        base6 = base64.b64encode(res_bytes)
        message = base6.decode('utf-8')

        await websocket.send(message)


def main(_argv):
    start_server = websockets.serve(sender, "127.0.0.1", 5678)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()


if __name__ == '__main__':
    try:

        app.run(main)
    except SystemExit:
        pass
