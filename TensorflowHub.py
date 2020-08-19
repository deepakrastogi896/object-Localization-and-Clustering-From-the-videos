import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_hub as hub
import os
import warnings  
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from PIL import Image
import time
class LoadHubModel(object):
  def __init__ (self,len_img,path1,nms_thres):
    self._len_img = len_img
    self._path1 = path1
    self._nms_thres =nms_thres
  def get_image_file_path(self,image_file_name):
    """returns the path of image file"""
    return self._path1 + image_file_name
  def get_images(self,n):
    """reads all the files from `../input/test` directory and returns paths for n files from top"""
    all_image_files = os.listdir(self._path1)
    # let's save all these image paths for later
    image_paths = list(map(self.get_image_file_path, all_image_files))
    # rather than using all, we will use a subset of these image paths for working on our model
    image_paths = image_paths[:n]
    return image_paths

  def get_image_id_from_path(self,image_path):
    """returns image id from image path"""
    return image_path.split(self._path1)[1].split('.jpg')[0]
  def ModelOperation(self,obj_sup,obj_getdf,images):
    image_paths=self.get_images(self._len_img)
    k = len(image_paths)
    predictions = []
    cropimage=[]
    k=-1
    cnt=0
    num_clu=0
    module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
    with tf.device('/device:GPU:0'):
      with tf.Graph().as_default():
        detector = hub.Module(module_handle)
        image_string_placeholder = tf.placeholder(tf.string)
        decoded_image = tf.image.decode_jpeg(image_string_placeholder)
        # Module accepts as input tensors of shape [1, height, width, 3], i.e. batch
        # of size 1 and type tf.float32.
        decoded_image_float = tf.image.convert_image_dtype(
            image=decoded_image, dtype=tf.float32)
        module_input = tf.expand_dims(decoded_image_float, 0)
        result = detector(module_input, as_dict=True)
        init_ops = [tf.global_variables_initializer(), tf.tables_initializer()]

        session = tf.Session()
        session.run(init_ops)
    with tf.device('/device:GPU:0'):
      for image_path in image_paths:
        k=k+1
        im = Image.fromarray(images[k])
        im.save("your_file.jpeg")
        with tf.gfile.Open(image_path, "rb") as binfile:
            image_string = binfile.read()

        inference_start_time = time.clock()
        result_out, image_out = session.run(
            [result, decoded_image],
            feed_dict={image_string_placeholder: image_string})
        df1=obj_getdf.get_prediction_string(result_out)
        # print(df1.shape)
        z1=obj_sup.nms(df1.values,self._nms_thres)
        z=df1.iloc[z1]
        z=z.reset_index()
        # print(z.shape)
        for j in range(len(z)):
          img=im
          img_class=z.iloc[j].Class_name
          img_xmax, img_ymax =images[k].shape[1],images[k].shape[0]
          bbox_x_max, bbox_x_min = z.Xmax[j] * img_xmax, z.Xmin[j] * img_xmax
          bbox_y_max ,bbox_y_min = z.Ymax[j] * img_ymax, z.Ymin[j] * img_ymax
          im1 = img.crop([bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max])
          cropimage.append(im1)
          cnt+=1
          if not os.path.exists('rois1/'+ str(z.iloc[j].Class_name)):
            os.makedirs('rois1/'+ str(z.iloc[j].Class_name))
            num_clu+=1  
          file =str(cnt)+'.jpg'
          try:
            im1.save('rois1/'+str(z.iloc[j].Class_name)+'/'+file)
          except Exception as e:
            print(e)
      return num_clu