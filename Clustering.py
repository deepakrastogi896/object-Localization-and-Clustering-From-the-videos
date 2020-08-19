from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import tqdm
import os
import pandas as pd
import numpy as np
class Cluster(object):
  def ApplyPcaAndClustering(self,array,z,num_clu):
    pca = PCA()
    pca.fit_transform(array)
    total=sum(pca.explained_variance_)
    k=0
    current_variance=0
    while current_variance/total < 0.99:
      current_variance += pca.explained_variance_[k]
      k=k+1
    pca=PCA(n_components=k)
    X_new=pca.fit_transform(array)
    km = AgglomerativeClustering(n_clusters=num_clu)
    pred = km.fit_predict(X_new)
    pred=pred.tolist()
    df = pd.DataFrame(pred, columns = ['cluster'])
    clabel=dict()
    for i in range(df.shape[0]):
       if not os.path.exists('clusterresult/'+str(df.iloc[i].cluster)):
            os.makedirs('clusterresult/'+str(df.iloc[i].cluster))
    for i in range(len(pred)):
      image=z[i]
      file = str(i)+'.jpg'
      image.save('clusterresult/'+ str(pred[i]) +'/'+file)