import numpy as np
from scipy.signal import decimate

def down_sample(img,n):
    img_d=decimate(img, n, n=2, ftype='iir',axis=0, zero_phase=True)
    img_d=decimate(img_d, n, n=2, ftype='iir',axis=1, zero_phase=True)
    return img_d.astype("uint8")

def xyrgb(img_arr):
    count = 0
    temp = np.zeros((5,img_arr.shape[0]*img_arr.shape[1]))
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            temp[:,count] = np.array([i,j,img_arr[i,j][0], img_arr[i,j][1], img_arr[i,j][2]])
            count +=1

    return temp

def normalise(xyrgb):
    inlessmeans = xyrgb- np.mean(xyrgb, axis=0)
    stds = np.std(xyrgb, axis=0)
    out = inlessmeans / stds
    return out

def colourlist(n):
    out = []
    r = np.random.randint(0,255)
    g = np.random.randint(0,255)
    b = np.random.randint(0,255)
    div = 256 / n
    for i in range(1,n+1):
        colour = np.array([(r+i*div)%256,(g+i*div)%256,(b+i*div)%256 ])
        out.append(colour)

    return out

def jaccard_index(img1, img2):
    """ Calculates the Jaccard index (IoU) measure for 2 mask representations
        of image foreground.
        img1, img2: A mask of each image, with 1's populating the foreground
        pixels, and 0s populating the background pixels.
    """

    # calculate where both,either, or no pixel is identified as foreground
    plus = img1 + img2

    # both as foreground
    i = np.where(plus==2,1, 0)

    # either as foreground
    j = np.where(plus != 0, 1, 0)

    # find intersection and union of masks
    intersection = i.sum()
    union = j.sum()

    return intersection / union

def generate_datasets()
