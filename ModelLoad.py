import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import cv2
import glob 
from keras.applications.resnet50 import preprocess_input
import os
import numpy as np
from PIL import Image
import time
l=[]
z=[]
class ModelLoader(object):
  def __init__ (self):
    self._model = ResNet50(weights='imagenet', include_top=False,pooling='avg')
  def extract_vector(self,folder):
    global l
    global z
    resnet_feature_list = []
    for sub_dir,dirs,files in os.walk(folder):
        for filename in files:
            img = Image.open(os.path.join(sub_dir, filename))
            #z.append(img)
            if img is not None:
                img1 = np.array(img)
                #print(img1.shape)
                if len(img1.shape)==3:
                    if img1.shape[2]==3:
                        z.append(img)
                        img1 = cv2.resize(img1,(224,224))
                        l.append(img1)
                        #print(img1.shape)
                        img1 = preprocess_input(np.expand_dims(img1.copy(), axis=0))
                        resnet_feature = self._model.predict(img1)
                        resnet_feature_list.append(resnet_feature.flatten())

    return np.array(resnet_feature_list),z
    