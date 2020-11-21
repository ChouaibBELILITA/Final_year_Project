import base64
import json
import asyncio
import websockets
import random
import os
import glob
import threading

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


import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs


flags.DEFINE_string('classes', './data/face.names', 'path to classes file')
flags.DEFINE_string('weights', './cheekpoint-chouaib/yolov3_train_12.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './Presidents.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID',
                    'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


#!/usr/bin/env python


#!/usr/bin/env python

# WS server that sends messages at random intervals



class DetecRecog(object):
    def __init__(self):
        self.physical_devices = tf.config.experimental.list_physical_devices(
            'GPU')
        if len(self.physical_devices) > 0:
            tf.config.experimental.set_memory_growth(
                self.physical_devices[0], True)

        if FLAGS.tiny:
            self.yolo = YoloV3Tiny(classes=FLAGS.num_classes)
        else:
            self.yolo = YoloV3(classes=FLAGS.num_classes)

        self.yolo.load_weights(FLAGS.weights)
        logging.info('weights loaded')
        self.vgg_face = tf.keras.models.load_model("vgg_model.h5")
        logging.info('vgg_model loaded')
        self.class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
        logging.info('classes loaded')

        self.times = []

        try:
            
            self.vid = cv2.VideoCapture(int(FLAGS.video))
            
        except:
            
            self.vid = cv2.VideoCapture(FLAGS.video)
            

        self.out = None
        self.outputed_image = None

        if FLAGS.output:
            # by default VideoCapture returns float instead of int
            self.width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(vid.get(cv2.CAP_PROP_FPS))
            self.codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
            self.out = cv2.VideoWriter(
                FLAGS.output, self.codec, self.fps, (self.width, self.height))

    def __del__(self):
        self.vid.release()
    
    def gen(self,camera):
        while True:
            image = next(camera.get_frame())

            yield(image)

    async def serve(self,websocket, path):
        camera = self
        i = 0
        while True:
            i = i+1
            data = next(self.gen(camera))
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

    def get_match(self, known_embeddings, candidate_embedding, thresh=0.3):

        minim = thresh
        person = "unknown"

        for key in known_embeddings:
            score = cosine(known_embeddings[key], candidate_embedding)
            if minim > score:
                minim = score
                person = key

        return person, minim

    def recognize(self, img, outputs, class_names, vgg_face):
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
                name, score = self.get_match(person_rep, embed, 0.30)

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
    
    def track(self,names,yolooutput):
        while True:
            img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            img_in = tf.expand_dims(img_in, 0)
            img_in = transform_images(img_in, FLAGS.size)

            t1 = time.time()
            boxes, scores, classes, nums = yolooutput
            ###############################################################################################################################################################################
            names = np.array(names)
            converted_boxes = convert_boxes(img, boxes[0])
            features = encoder(img, converted_boxes)    
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(converted_boxes, scores[0], names, features)]
            
            #initialize color map
            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            # run non-maxima suppresion
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]        

            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                class_name = track.get_class()
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(img, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                
            ### UNCOMMENT BELOW IF YOU WANT CONSTANTLY CHANGING YOLO DETECTIONS TO BE SHOWN ON SCREEN
            #for det in detections:
            #    bbox = det.to_tlbr() 
            #    cv2.rectangle(img,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
            
            # print fps on screen 
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
            cv2.imshow('output', img)
            if FLAGS.output:
                out.write(img)
                frame_index = frame_index + 1
                list_file.write(str(frame_index)+' ')
                if len(converted_boxes) != 0:
                    for i in range(0,len(converted_boxes)):
                        list_file.write(str(converted_boxes[i][0]) + ' '+str(converted_boxes[i][1]) + ' '+str(converted_boxes[i][2]) + ' '+str(converted_boxes[i][3]) + ' ')
                list_file.write('\n')

            # press q to quit
            if cv2.waitKey(1) == ord('q'):
                break
 




        ################################################################################################################################################################################

    def tmr(self,i):
        start_server = websockets.serve(self.serve, "127.0.0.1", 5678+i)
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()
    
    def thr(self,i):
        # we need to create a new loop for the thread, and set it as the 'default'
        # loop that will be returned by calls to asyncio.get_event_loop() from this
        # thread.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        loop.run_until_complete(self.tmr(i))
        loop.close()
    def run(self):
        t=threading.Thread(target=self.thr, args=(0,), daemon=True)
        t.start()
        
        
        while True:
            
            _, img = self.vid.read()
            

            if img is None:
                logging.warning("Empty Frame")
                time.sleep(0.1)
                continue
            
            
            img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_in = tf.expand_dims(img_in, 0)
            img_in = transform_images(img_in, FLAGS.size)

            t1 = time.time()
            boxes, scores, classes, nums = self.yolo.predict(img_in)
            t2 = time.time()
            self.times.append(t2-t1)
            self.times = self.times[-20:]

            img = self.recognize(img, (boxes, scores, classes, nums),
                                 self.class_names, self.vgg_face)
            img = cv2.putText(img, "Time: {:.2f}ms".format(sum(self.times)/len(self.times)*1000), (0, 30),
                              cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
            if FLAGS.output:
                self.out.write(img)
            cv2.imshow('output', img)
            self.outputed_image = img
            if cv2.waitKey(1) == ord('q'):
                break

    def get_frame(self):
        while True:
            ret, jpeg = cv2.imencode('.jpg', self.outputed_image)
            base6 = base64.b64encode(jpeg.tobytes())
            yield base6.decode('utf-8')



def main(_argv):
    prog = DetecRecog()
    prog.run()

print('ccccccddddddddd')
if __name__ == '__main__':
    print('aaabbb')
    try:

        app.run(main)
    except SystemExit:
        pass