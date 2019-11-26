from scipy.spatial.distance import cdist
from sklearn.cluster import SpectralClustering
import numpy as np

def graph_cut(img,K):
    width=img.shape[1]
    height=img.shape[0]
    xv, yv = np.meshgrid(np.arange(0, width), np.arange(0, height))
    shape=(1,-1)
    a=np.concatenate((img[:,:,0].reshape(shape),img[:,:,1].reshape(shape),img[:,:,2].reshape(shape))).T
    a=np.concatenate((yv.reshape(shape),xv.reshape(shape),img[:,:,0].reshape(shape),img[:,:,1].reshape(shape),img[:,:,2].reshape(shape))).T
    V=np.var(a, axis =0)
    V[-3:]*=1.3
    s=width*height
    W=np.exp(-(cdist(a,a,'seuclidean', V=V)/20))
    D=(np.sum(W, axis=1)) * np.eye(s)
    clustering = SpectralClustering(n_clusters=K,
            assign_labels="discretize",
            random_state=0,affinity="precomputed").fit(W)
    return clustering.labels_

def color_from_label(labels, img, k):
    shape=(1,-1)
    a=np.concatenate((img[:,:,0].reshape(shape),img[:,:,1].reshape(shape),img[:,:,2].reshape(shape))).T
    index_mask_k=np.where(labels==k)
    color = np.mean(a[index_mask_k][:,-3:], axis=0)
    return color