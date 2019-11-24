import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.signal import decimate
from utils import down_sample, normalise, colourlist

K=10
N=4

def RunKMeans(features, img_shape, clusters=K):

    """
        Given a feature representation and the original
        shape of an image, run K-Means segmentation and return
        an array in the original image shape with each pixel populated
        by a label for its cluster.
    """

    # generate kmeans object and fit to features
    km = KMeans(clusters)
    km.fit(features)

    # get cluster labels for each pixel, and reshape to image size
    labels = km.labels_.reshape(img_shape)

    return labels

def FindForegroundCluster(labels, mask, clusters=K):

    """
        Given a representation of an images as cluster labels,
        and a binary mask of the foreground of that image, find
        cluster with highest Jaccard Index (IoU score), and return
        that score
    """

    js = []
    for j in range(clusters):

        # compute jaccard index for each cluster
        a = np.where(labels==j,1,0)
        js.append(jaccard_index(a,mask))

    return max(js)

def main():
    data_path = os.getcwd() + "/data/JPEGImages/480p/"
    anno_path = os.getcwd() + "/data/Annotations/480p/"

    """img_paths = []
    mask_paths = []

    for root,_,paths  in os.walk(data_path):
        img_paths += [root+path for path in paths]

    for root,_,paths in os.walk(anno_path):
        mask_paths"""

    vids_img = []
    for thing,_,_ in os.walk(data_path):
        vids_img.append(thing)

    vids_mask = []
    for thing,_,_ in os.walk(mask_path):
        #print(thing)
        vids_mask.append(thing)

    imgs = []
    masks = []

    for i in range(len(vids_img[1:])):
        di_img = vids_img[i+1]
        di_mask = vids_mask[i+1]
        temp1 = []
        temp2 = []
        for _,_,paths in os.walk(di_img):
            for path in paths:
                #print(path)
                if di_img+"/"+path == bear_img_path + "00077.jpg":
                    #print(path)
                    pass
                else:
                    #print(path)
                    img = np.asarray(Image.open(di_img+"/"+path))
                    mask = np.asarray(Image.open(di_mask+"/"+path[:-3]+"png"))
                    temp1.append(img)
                    temp2.append(mask)

        imgs += temp1
        masks +=  temp2
