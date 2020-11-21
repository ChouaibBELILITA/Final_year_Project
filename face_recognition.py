import os
import glob
import sys
import cv2
import matplotlib.pyplot as plt
import dlib

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import ZeroPadding2D,Convolution2D,MaxPooling2D
from tensorflow.keras.layers import Dense,Dropout,Softmax,Flatten,Activation,BatchNormalization
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow.keras.backend as K

from scipy.spatial.distance import cosine



import time
from absl import app, flags, logging
from absl.flags import FLAGS
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs
np.set_printoptions(threshold=sys.maxsize)

flags.DEFINE_string('classes', './data/face.names', 'path to classes file')
flags.DEFINE_string('weights', './cheekpoint-chouaib/yolov3_train_12.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', 'people11.jpg', 'path to input image')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('output', './output.jpg', 'path to output image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')










def get_match(known_embeddings, candidate_embedding, thresh=0.5):
    min = thresh
    person="unknown"
    
    for key in known_embeddings:
      score = cosine(known_embeddings[key], candidate_embedding)
      if min >score :
        min=score
        person=key
    return person,min


def recognize(img, outputs, class_names):
    person_rep=dict()
    person_names=["angelamerkel","jinping","trump"]
    for person in person_names:
        embed=np.loadtxt(person+".txt")
        person_rep[person]=embed

    vgg_face=tf.keras.models.load_model("vgg_model.h5")
    logging.info('vgg_model loaded')



    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        if class_names[int(classes[i])] == "face":
            print("faceaaaaaaaaaaaaaaaaaaaaaaaaa")
            img_crop=img[x1y1[1]:x2y2[1],x1y1[0]:x2y2[0]]
            
            crop_img=img_to_array(img_crop)
            crop_img=np.expand_dims(crop_img,axis=0)
            crop_img=preprocess_input(crop_img)


            img_encode=vgg_face(transform_images(crop_img, 224))
            embed=K.eval(img_encode)
            name, score= get_match(person_rep, embed, 0.3)

            img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
            img = cv2.putText(img, '{} {:.4f}'.format(
                name, score),
                x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        else:
            img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
            img = cv2.putText(img, '{} {:.4f}'.format(
                class_names[int(classes[i])], objectness[i]),
                x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img





def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('yolo weights loaded')



    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    if FLAGS.tfrecord:
        dataset = load_tfrecord_dataset(
            FLAGS.tfrecord, FLAGS.classes, FLAGS.size)
        dataset = dataset.shuffle(512)
        img_raw, _label = next(iter(dataset.take(1)))
    else:
        img_raw = tf.image.decode_image(
            open(FLAGS.image, 'rb').read(), channels=3)

    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, FLAGS.size)

    t1 = time.time()
    boxes, scores, classes, nums = yolo(img)
    t2 = time.time()


    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)

    img = recognize(img, (boxes, scores, classes, nums), class_names)
    cv2.imwrite(FLAGS.output, img)
    logging.info('output saved to: {}'.format(FLAGS.output))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass












