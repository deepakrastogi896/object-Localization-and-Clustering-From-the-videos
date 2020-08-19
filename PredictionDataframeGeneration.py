import pandas as pd
import tensorflow as tf
class GetDataFrame(object):
  def __init__ (self,min_thresh):
    self._min_thresh = min_thresh
  def get_prediction_string(self,result):
    with tf.device('/device:GPU:0'):
        """from each result, generates the complete prediction string in the format {Label Confidence XMin YMin XMax YMax},{...} based on submission file."""
        df = pd.DataFrame(columns=['Ymin','Xmin','Ymax', 'Xmax','Score','Label','Class_name'])
        min_score=self._min_thresh
        for i in range(result['detection_boxes'].shape[0]):
           if (result["detection_scores"][i]) >= min_score:
              df.loc[i]= tuple(result['detection_boxes'][i])+(result["detection_scores"][i],)+(result["detection_class_labels"][i],)+(result["detection_class_entities"][i],)
        return df