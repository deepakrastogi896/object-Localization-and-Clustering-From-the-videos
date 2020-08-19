from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps
import numpy as np
import cv2
import glob
import os
class ImageLoader(object):
  def LoadImages(self,images,path1):
    for f in glob.iglob(path1+"*"):
      images.append(np.asarray(Image.open(f)))
    return images