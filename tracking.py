import base64
import json
import asyncio
import websockets
import random
import os
import sys
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
from tools import generate_detections as gdet
from deep_sort.tracker import Tracker
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection

from yolov3_tf2.models import (
    YoloV3
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, convert_boxes


from apscheduler.schedulers.blocking import BlockingScheduler

# flags.DEFINE_string('classes', './data/face.names', 'path to classes file')
# flags.DEFINE_string('weights', './checkpoints/yolov3_train_10.tf',
                    # 'path to weights file')
# flags.DEFINE_integer('size', 416, 'resize images to')
# flags.DEFINE_string('video', './track.mp4',
#                     'path to video file or number for webcam)')
# flags.DEFINE_string('output', None, 'path to output video')
# flags.DEFINE_string('output_format', 'XVID',
#                     'codec used in VideoWriter when saving video to file')
# flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


#!/usr/bin/env python


#!/usr/bin/env python

# WS server that sends messages at random intervals
from datetime import datetime
from django.core.files import File
current_file = os.path.dirname(__file__)




def get_time () :
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M")
    return dt_string
class DetecRecog(object):
    def __init__(self):
        from detections.models import Person,Video,Detections,Camera
        self.per_var = Person
        self.video_var = Video
        self.detection_var = Detections
        self.camera_var = Camera
        self.physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(self.physical_devices) > 0:
            tf.config.experimental.set_memory_growth(
                self.physical_devices[0], True)
        self.yolo = YoloV3(classes=80)
        
        self.yolo.load_weights(current_file+'/cheekpoint-chouaib/yolov3_train_12.tf')
        logging.info('weights loaded')
        self.vgg_face = tf.keras.models.load_model(current_file+'/vgg_model.h5')
        logging.info('vgg_model loaded')
        self.class_names = [c.strip() for c in open(current_file+'/data/face.names').readlines()]
        logging.info('classes loaded')

        self.times = []

        # Definition of the parameters
        self.max_cosine_distance = 0.5
        self.nn_budget = None
        self.nms_max_overlap = 1.0
        
        #initialize deep sort
        self.model_filename = current_file+'/model_data/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(self.model_filename, batch_size=1)
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        self.tracker = Tracker(self.metric)
        try:
            
            self.vid = cv2.VideoCapture(int('FLAGS.video'))
            
        except:
            self.vid = cv2.VideoCapture(current_file+'/DonaldTrump.mp4')
            

        self.out = None
        """self.output_video = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))"""
        self.outputed_image = None
        self.person_dict = {}
        self.output_data =  []
        self.output_path=get_time()+".mp4"
        self.detection = set()
        if self.output_path: #FLAGS.output:
            self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.vid.get(cv2.CAP_PROP_FPS))
            self.codec = cv2.VideoWriter_fourcc(*'H264')
            self.out = cv2.VideoWriter(
                self.output_path, self.codec, self.fps, (self.width, self.height))
        else : 
            self.fps = 0.0
        
        t=threading.Thread(target = self.sheduling,daemon=True)
        t.start()

    def __del__(self):
        self.vid.release()
    
    def sheduling(self):
        scheduler = BlockingScheduler()
        scheduler.add_job(self.save_video, 'interval', hours=0.016)
        scheduler.start()
        
    def save_video(self):

        if self.out:
            self.out.release()
            video = self.video_var(camera = self.camera_var.objects.get(camid=0))
            video.videopath.save(self.output_path , File(open(os.path.join(".",self.output_path),"rb")))
            print(self.detection)
            for person in self.detection :
                if person != None : 
                    print(person)
                    det = self.detection_var(video=video , person=person)
                    det.save()
            self.detection = None
            self.out=None
        self.detection = set()
        self.output_path=get_time()+".mp4"

        if self.output_path: #FLAGS.output:
            print(self.output_path)
            self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)+0.5)
            self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)+0.5)
            self.fps = int(self.vid.get(cv2.CAP_PROP_FPS))
            self.codec = cv2.VideoWriter_fourcc(*'H264')
            self.out = cv2.VideoWriter(
                self.output_path, self.codec, self.fps, (self.width, self.height))
        else : 
            self.fps = 0.0

    def gen(self):
        while True:
            image = next(self.get_frame())

            yield(image)
    def database_search(self, vecteur) :
        pass
    async def serve(self,websocket, path):
        
        camera = self
        i = 0
        while True:
            i = i+1
            data = next(self.gen())
            js =  {'data': data, "infos": self.output_data  }
            res_bytes = json.dumps(js).encode('utf-8')
            base6 = base64.b64encode(res_bytes)
            message = base6.decode('utf-8')

            await websocket.send(message)

    def get_match(self, candidate_embedding, thresh=0.3):
        result = self.per_var.objects.get_closest_user(candidate_embedding)
        minim = thresh
        person = "unknown"
        cos_var = result[1]

        if cos_var <= 0.3:
            if self.detection != None : 
                self.detection.add(result[0])
            return result
        else :
            return None,cos_var
        """for key in known_embeddings:
            score = cosine(known_embeddings[key], candidate_embedding)
            if minim > score:
                minim = score
                person = key"""
                
        #return person, minim

    def recognize(self, img, outputs, vgg_face):
        output_names = []
        output_persons={}
        output_scores=[]
        """person_rep = dict()
        person_names = ["angelamerkel", "jinping", "trump"]
        for person in person_names:
            embed = np.loadtxt(current_file+'/'+person+".txt")
            person_rep[person] = embed"""
        boxes, objectness, classes, nums = outputs
        boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
        wh = np.flip(img.shape[0:2])
        outputed_data = []
        for i in range(nums):
            x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
            x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
            if self.class_names[int(classes[i])] == "face":

                img_crop = img[x1y1[1]:x2y2[1], x1y1[0]:x2y2[0]]
                crop_img = img_to_array(img_crop)
                crop_img = np.expand_dims(crop_img, axis=0)
                crop_img = preprocess_input(crop_img)

                img_encode = vgg_face(transform_images(crop_img, 224))
                embed = K.eval(img_encode)
                person, score = self.get_match(embed, 0.3)
               
                
                if person is None :
                    output_names.append('Unknown')
                else :
                    output_names.append(person.familyName)
                    output_persons[person.familyName] = person
                    """person_data = {
                    "key": person.id,
                    "name": person.familyName+" "+person.firstName,
                    "age": person.age,
                    "address": person.address,
                    }
                    outputed_data.append(person_data)""" 
                output_scores.append(score) 
                 
        #self.output_data = outputed_data              
        return (output_persons,output_names,output_scores)
    
    def track(self, img, yolooutput, recognizeoutput):
        
        #img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        #img_in = tf.expand_dims(img_in, 0)
        #img_in = transform_images(img_in, 416) #416 =size

        t1 = time.time()

        boxes, detect_scores, classes, nums = yolooutput
        persons,names,scores=recognizeoutput
        self.person_dict = dict(self.person_dict, **persons)
        ###############################################################################################################################################################################
        names = np.array(names)
        converted_boxes = convert_boxes(img, boxes[0])
        features = self.encoder(img, converted_boxes)    
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(converted_boxes, scores, names, features)] #if there is an error here try removing "[0]"
        
        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima suppresion
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]        

        # Call the tracker
        self.tracker.predict()
        self.tracker.update(detections)
        output_data = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            if class_name in self.person_dict :
                person = self.person_dict[class_name]
                if person :
                    person_data = {
                            "key": person.id,
                            "name": person.familyName+" "+person.firstName,
                            "age": person.age,
                            "address": person.address,
                            }
                    output_data.append(person_data)
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(img, class_name,(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            

        """if self.output_path:
            self.out.write(img)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(converted_boxes) != 0:
                for i in range(0,len(converted_boxes)):
                    list_file.write(str(converted_boxes[i][0]) + ' '+str(converted_boxes[i][1]) + ' '+str(converted_boxes[i][2]) + ' '+str(converted_boxes[i][3]) + ' ')
            list_file.write('\n')"""
        self.output_data = output_data
        return img
        # press q to quit
 




        ################################################################################################################################################################################

    def tmr(self,i):
        start_server = websockets.serve(self.serve, "127.0.0.1", 5678+i)
        print("pranty kach haja")
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()
    
    def thr(self,i):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        loop.run_until_complete(self.tmr(i))
        loop.close()
    def run(self):
        t=threading.Thread(target=self.thr, args=(0,), daemon=True)
        t.start()
        while True:
            
            _, img1 = self.vid.read()
            

            if img1 is None:
                logging.warning("Empty Frame")
                time.sleep(0.1)
                continue
            
            
            img_in = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img_in = tf.expand_dims(img_in, 0)
            img_in = transform_images(img_in, 416)

            t1 = time.time()
            yoloOutputs = self.yolo.predict(img_in)
            recognizeOutputs = self.recognize(img1, yoloOutputs, self.vgg_face)
            img=self.track(img1, yoloOutputs, recognizeOutputs)
            t2 = time.time()
            self.times.append(t2-t1)
            self.times = self.times[-20:]


            
            ########################################################################################
            img = cv2.putText(img, "Time: {:.2f}ms".format(sum(self.times)/len(self.times)*1000), (0, 30),
                              cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
            if self.out :
                self.out.write(img)
            self.outputed_image = img
            if cv2.waitKey(1) == ord('q'):
                break
        t.join()
    def get_frame(self):
        while True:
            ret, jpeg = cv2.imencode('.jpg', self.outputed_image)
            base6 = base64.b64encode(jpeg.tobytes())
            yield base6.decode('utf-8')



def main(_argv):
    prog = DetecRecog()
    prog.run()


def run():
    try:
        app.run(main)
    except SystemExit :
        pass