# -*- coding: utf-8 -*-
"""UOD.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VIQ0ZHkAvoCaAwayJyknsrfFcmFkmrXe
"""

#!pip install ffmpy

from google.colab import drive
drive.mount('downloadimage')

#!rm -r /content/video/frame

#!rm -r /content/clusterresult

#!mkdir frames

#!mkdir video

#For Videos
from frame_extraction import FrameExtraction
folder_name="frame"
path_dir="/content/video/"
fps = 1/5
frames=FrameExtraction(folder_name)
frames.videos_to_frame(path_dir,fps)
path =path_dir+folder_name
path1=path+"/"

#For Images
#path1='/content/frames/'

from ImageLoaderFromFrame import ImageLoader
List=[]
img_loader=ImageLoader()
images = img_loader.LoadImages(List,path1)

len(images)

#select Min_Score and NMS_Threshold
from PredictionDataframeGeneration import GetDataFrame
from Suppression import NonMaxSuppression
from TensorflowHub import LoadHubModel
obj_sup=NonMaxSuppression()
min_scores=0.01
obj_getdf = GetDataFrame(min_scores)
t=len(images)
nms_threshold=0.6
obj_tfhub = LoadHubModel(t,path1,nms_threshold)
num_clu=obj_tfhub.ModelOperation(obj_sup,obj_getdf,images)

#Clustering

from ModelLoad import ModelLoader
roi_dir="/content/rois1"
load_model=ModelLoader()
array,z =load_model.extract_vector(roi_dir)

#you Can select number of cluster
from Clustering import Cluster
clu_obj=Cluster()
num_cluster=num_clu+int(num_clu*0.1)
clu_obj.ApplyPcaAndClustering(array,z,num_cluster)

#!zip -r ./clusterresult.zip ./clusterresult/

