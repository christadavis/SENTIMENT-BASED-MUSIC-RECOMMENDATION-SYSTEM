from __future__ import absolute_import

import torch
import numpy as np
import os, sys, math
import cv2, time
from detector.data import cfg_resnet18
from detector.layers.functions.prior_box import PriorBox
from detector.utils.nms.py_cpu_nms import py_cpu_nms
from detector.utils.box_utils import decode, decode_landm
from tensorflow.keras.models import Model,Sequential, load_model, model_from_json

import tensorflow as tf
#from tensorflow.compat.v1.keras.backend import set_session 
from tensorflow.python.keras import backend as K
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

sess=tf.compat.v1.Session(config=config)
K.set_session(sess)
import pandas as pd

class RetinaFace(object):
    
    def __init__(self, jit_file = "./detector/models/resnet18.zip", cpu = True, top = 1500, conf = 0.6):
        
        self.device = torch.device("cpu" if cpu else "cuda")
        self.net = torch.jit.load(jit_file)
        
        self.net = self.net.to(self.device)
        self.cfg = cfg_resnet18
        self.nms_threshold = 0.3
        self.confidence_threshold = conf
        self.top_k = top
        self.priors = None
        
        
    def detect_faces(self, input_image):
        
        img = np.float32(input_image)
        img -= (104, 117, 123)

        img_height, img_width, _ = input_image.shape
        scale = torch.Tensor([input_image.shape[1], input_image.shape[0], input_image.shape[1], input_image.shape[0]])
        
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)
        
        loc, conf, landms = self.net(img)

        if self.priors is None:
            priorbox = PriorBox(self.cfg, image_size=(img_height, img_width))
            priors = priorbox.forward()
            priors = priors.to(self.device)
            prior_data = priors.data

        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1
        landms = landms.cpu().numpy()
        
        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]
        dets = np.concatenate((dets, landms), axis=1)
        
        return dets
    
    def face_crop(self, input_image, dets, resolution=160, flip = False, align= False):
        
        img_height, img_width, _ = input_image.shape
        crops = np.zeros((0, resolution, resolution, 3), dtype=np.float32)
        crops_flips = np.zeros((0, resolution, resolution, 3), dtype=np.float32)
        t = time.time()
        for detection in dets:
            
            d = list(detection)
            d = [int(math.ceil(max(box, 0))) for box in d]
            
            x = int(( d[0] + d[2] ) / 2)
            y = int(( d[1] + d[3] ) / 2)
            
            w = int(d[2] - d[0])
            h = int(d[3] - d[1])
            
            if align:
                _direction = np.degrees(np.arctan2((d[8] - d[6]),(d[7] - d[5])))
                eye_center = ((d[7] + d[5])//2, (d[8] + d[6])//2 )
                transformation_matrix = cv2.getRotationMatrix2D(eye_center, _direction, 1)
                rotated_image = cv2.warpAffine(input_image, transformation_matrix, (img_width, img_height))
                crop = rotated_image[d[1]:d[1]+h,d[0]:d[0]+w,:]
            else:
                crop = input_image[d[1]:d[1]+h,d[0]:d[0]+w,:]

            if 0 in crop.shape:
                continue
            crop = cv2.resize(crop, (resolution, resolution))
            if flip:
                flip_crop = cv2.flip(crop, 1)
                flip_crop = np.expand_dims(flip_crop, axis=0).astype(np.float32)
                crops_flips = np.concatenate((crops_flips, flip_crop), axis=0)
                
            crop = np.expand_dims(crop, axis=0).astype(np.float32)
            crops = np.concatenate((crops, crop), axis=0)
            
        if flip:
            return crops, crops_flips
        else:
            return crops
        

def predict_songs(mood):

    mood_music = pd.read_csv("data_moods.csv")
    mood_music = mood_music[['name','artist','mood']]

    if(mood==0 or mood==1 or mood==2 ):
        #for angery,disgust,fear
        filter1=mood_music['mood']=='Calm'
        f1=mood_music.where(filter1)
        f1=f1.dropna()
        f2 =f1.sample(n=5)
        f2.reset_index(inplace=True)
        print(f2.head())
    if(mood==3 or mood==4):
        #for happy, neutral
        filter1=mood_music['mood']=='Happy'
        f1=mood_music.where(filter1)
        f1=f1.dropna()
        f2 =f1.sample(n=5)
        f2.reset_index(inplace=True)
        print(f2.head())
    if(mood==5):
        #for Sad
        filter1=mood_music['mood']=='Sad'
        f1=mood_music.where(filter1)
        f1=f1.dropna()
        f2 =f1.sample(n=5)
        f2.reset_index(inplace=True)
        print(f2.head())
    if(mood==6):
        #for surprise
        filter1=mood_music['mood']=='Energetic'
        f1=mood_music.where(filter1)
        f1=f1.dropna()
        f2 =f1.sample(n=5)
        f2.reset_index(inplace=True)
        print(f2.head())
        
    
if __name__ == "__main__":

    INPUT_SIZE = (224, 224)
    face = RetinaFace("./detector/models/mobilenet0.25.zip", True, 100)
    cap = cv2.VideoCapture(0)

    idx_to_class={0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Neutral', 5: 'Sadness', 6: 'Surprise'}

    model = load_model('./mobilenet_7.h5')

    start = time.time()

    emotions = list()
    emotion_placeholder = "analysing"

    while(cap.isOpened()):
        ret, img = cap.read()
        detections = face.detect_faces(img)
        face_crops = face.face_crop(img, detections, 224)

        for dets, crop in zip(detections, face_crops):
                        
            x1, y1, x2, y2 = tuple([int(x) for x in dets[:4]])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            inp = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            inp[..., 0] -= 103.939
            inp[..., 1] -= 116.779
            inp[..., 2] -= 123.68
            inp = np.expand_dims(inp, axis=0)
            
            scores = model.predict(inp)[0]
            label = idx_to_class[np.argmax(scores)]
            emotions.append(np.argmax(scores))

        cv2.rectangle(img, (x1, y1), (x1 + (len(emotion_placeholder)) * 15, 
                        y1 - 20) , (0,255,0), -1, cv2.LINE_AA)
        cv2.putText(img, emotion_placeholder , (x1, y1), 
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            
        end = time.time()
        # every 10 seconds
        if int(end - start) > 10:
            pred_emotion = max(emotions)
            emotions = list()
            start = end
            emotion_placeholder =  idx_to_class[pred_emotion]
            predict_songs(pred_emotion)

        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
        cv2.imshow("Result", img)
        if cv2.waitKey(1) == ord('q'):
            break 
    
    