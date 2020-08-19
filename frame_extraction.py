import os
class FrameExtraction(object):
  def __init__(self,dir_name):
    self._dir_name=dir_name
  def videos_to_frame(self,videopath_dir,fps):
    path = os.path.join(videopath_dir,self._dir_name)
    os.mkdir(path)
    path1=path+"/"
    ch='a'
    for filename in os.listdir(videopath_dir): #or .avi, .mpeg, whatever.
      if (filename.endswith(".mp4")):
        pathx =videopath_dir+filename
        ch=chr(ord(ch) + 2)
        os.system("ffmpeg -i {0} -filter:v fps=fps={1} {2}{3}_%d.jpg".format(pathx,fps,path1,ch))
      else:
        continue