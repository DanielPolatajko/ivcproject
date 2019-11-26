import tqdm
from scipy.spatial.distance import cdist
import numpy as np

def means_shift(img,w,iterations, colors_weight):
    width=img.shape[1]
    height=img.shape[0]
    xv, yv = np.meshgrid(np.arange(0, width), np.arange(0, height))
    shape=(1,-1)
    a=np.concatenate((yv.reshape(shape),xv.reshape(shape),img[:,:,0].reshape(shape),img[:,:,1].reshape(shape),img[:,:,2].reshape(shape))).T
    V=np.var(a, axis =0)
    V[-3:]*=1/colors_weight
    num_pixels=a.shape[0]
    means = a.copy().astype("float64")
    log_0 = tqdm.tqdm(total=iterations, desc='Batch', position=0, ncols=80)
    stop=np.zeros(num_pixels)
    for iteration in range(iterations):
        means_last=means.copy()
        distances = cdist(a, means, 'seuclidean', V=V)
        #distances = cdist(a, means,)
        for i in range(num_pixels):
            if stop[i]==False:
                neighbourhood = means[np.argpartition(distances[i], w)][:w]
                means[i,:]=np.mean(neighbourhood, axis=0)
        stop = np.sum(means_last==means,axis=1)==means.shape[1]
        log_0.set_description_str(f"{stop.mean():0.3}")
        log_0.update(1)
    return means, V


def cluster_means(means, height, width, bins,k ):
    h1=np.histogramdd(means, bins=bins)
    bin_edges = h1[1]
    scaled_edges = h1[1].copy()
    scaled_means= means.copy()

    for i in range(5):
        scaled_means[:,i]= (means[:,i]-scaled_edges[i][0])/(scaled_edges[i][-1]-scaled_edges[i][0])*bins + 0.01
        scaled_edges[i]=(scaled_edges[i]-scaled_edges[i][0])/(scaled_edges[i][-1]-scaled_edges[i][0])*bins + 0.01

    h2=np.histogramdd(scaled_means, bins=scaled_edges)

    means_in_bins=np.digitize(scaled_means, h2[1][0][:50])
    hist_shape=h2[0].shape
    resh_h=h2[0].reshape(-1)
    greatest_1d_indices=  np.argpartition(resh_h, -k)[-k:]
    greatest_5d_indices = [np.unravel_index(i, hist_shape) for i in greatest_1d_indices]
    cluster_centers = []
    for i in range(k):
        gi = greatest_5d_indices[i]
        cluster_centers.append(
            np.mean(np.asarray([[h1[1][i][gi[i]] for i in range(5)], [h1[1][i][gi[i] + 1] for i in range(5)]]),
                    axis=0).astype("uint8"))
    cluster_weights = np.asarray([resh_h[i] for i in greatest_1d_indices])
    cluster_centers = np.asarray(cluster_centers)
    sort_idx = np.argsort(cluster_weights)[::-1]
    cluster_weights = cluster_weights[sort_idx]
    cluster_centers = cluster_centers[sort_idx]
    greatest_bins = np.asarray(greatest_5d_indices)[sort_idx]

    return greatest_bins,  means_in_bins , cluster_weights , cluster_centers

def lump_clusters(cluster_weights,cluster_centers,V,reach_constant=0.1):
    k= cluster_weights.shape[0]
    lookup=np.arange(k)
    lumped_centers=[]
    lumped_weights=[]
    for i in range(k):
        if cluster_weights[i]>0:
            cluster_absorbed=False
            for j in range(len(lumped_centers)):
                lumped_center=lumped_centers[j]
                lumped_weight=lumped_weights[j]
                scaled_distance = np.sum(((cluster_centers[i]-lumped_center)/np.sqrt(V))**2)**0.5
                reach = (cluster_weights[i]**0.5+ lumped_weight**0.5)
                if scaled_distance <reach*reach_constant:
                    lumped_centers[j]=(lumped_center*lumped_weight + cluster_centers[i]*cluster_weights[i])/(cluster_weights[i]+lumped_weight)
                    lumped_weights[j]=(cluster_weights[i]+lumped_weight)
                    cluster_absorbed=True
                    #print(f"{i} in {j}, {cluster_centers[i]} , {cluster_weights[i]}")
                    lookup[i]=j
                    break
            if not cluster_absorbed :
                lumped_centers.append(cluster_centers[i])
                lumped_weights.append(cluster_weights[i])
                #print(f"add {i} , {cluster_centers[i]} , {cluster_weights[i]}")
                lookup[i]=len(lumped_centers)-1
            res = np.hstack((np.asarray(lumped_centers),np.asarray(lumped_weights).reshape(-1,1)))
    lumped_centers=np.asarray(lumped_centers)
    lumped_weights=np.asarray(lumped_weights)
    return lumped_weights, lumped_centers,lookup


def make_labels(height,width,greatest_5d_indices,means_in_bins,lookup, lumped_centers):

    labels=np.ones((height,width)).reshape(-1)
    for i in range(height*width):
        match = (greatest_5d_indices==means_in_bins[i]-1).all(axis=1)
        if  not match.any():
            print (i , means_in_bins[i])
        labels[i] = np.argmax(match)
    labels_lookup= labels.copy()
    labels_lookup=lookup[labels_lookup.astype("int")]
    coloured_lookup=lumped_centers[labels_lookup.astype("int")][:,2:].reshape(height,width,3).astype("uint8")
    return labels_lookup, coloured_lookup

