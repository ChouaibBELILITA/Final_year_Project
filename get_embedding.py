import os
import glob
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
    YoloV3
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs



"""flags.DEFINE_string('classes', './data/face.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3_train_10.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', 'summit.jpg', 'path to input image')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('output', './output.jpg', 'path to output image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')"""


current_file = os.path.dirname(__file__)
def get_embedding(img_raw) :
    print(current_file)
    checkpoint_path = os.path.join(os.path.join(current_file,"checkpoints"),"yolov3_train_10.tf")
    data_path = os.path.join(os.path.join(current_file,"data"),"face.names")
    vgg_path = os.path.join(current_file,"vgg_model.h5")
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    image = tf.image.decode_image(img_raw, channels=3)
    img_raw = image
    img = tf.expand_dims(image, 0)
    img = transform_images(img, 416)
    
    yolo = YoloV3(classes=80)

    yolo.load_weights(checkpoint_path).expect_partial()
    class_names = [c.strip() for c in open(data_path).readlines()]

    boxes, scores, classes, nums = yolo(img)
    img = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2BGR)

    vgg_face=tf.keras.models.load_model(vgg_path)
    objectness = scores
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        embed = None
        if class_names[int(classes[i])] == "face":
            print("faaaaaaaaaaaaaaaaaaaaaaaaaceeeeeeeeeeeeeeee")
            img_crop=img[x1y1[1]:x2y2[1],x1y1[0]:x2y2[0]]
            
            crop_img=img_to_array(img_crop)
            crop_img=np.expand_dims(crop_img,axis=0)
            crop_img=preprocess_input(crop_img)
            img_encode=vgg_face(transform_images(crop_img, 224))
            embed=K.eval(img_encode)
            print("PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP")
            print(embed[0])
            return embed[0]
"""def embedding(img_raw):   
    try:
        app.config["img_raw"]=img_raw
        app.run(get_embedding)
        return app.config["output"]
    except SystemExit:
        pass"""

    












