import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.signal import decimate
from utils import down_sample, xyrgb, colourlist, generate_dataset_unsupervised, jaccard_index, normalise
import time
from random import shuffle

K=2
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

    imgs,masks = generate_dataset_unsupervised(data_path, anno_path,hsv=False)

    shuffle(imgs)
    shuffle(masks)

    j_scores = []
    print("There are " + str(len(imgs)) + " images in the dataset.")

    for i in range(len(imgs)):
        a = time.time()
        print(i)
        img = imgs[i]
        mask = masks[i]

        features = normalise(xyrgb(img).T)

        labels = RunKMeans(features, mask.shape, K)

        best_label = FindForegroundCluster(labels, mask, K)

        binary_fore = np.where(labels == best_label, 1, 0)
        binary_mask = np.where(mask >0 , 1 , 0)

        j = jaccard_index(binary_fore,binary_mask)

        print("This runs jscore" + str(j))

        j_scores.append(j)

        b = time.time()

        print("Running mean j-score" + str(np.mean(j_scores)))

        print(b-a)

    print("The mean Jaccard index across the data was " + str(np.mean(j_scores)))

    return j_scores

if __name__ == "__main__":
    main()
