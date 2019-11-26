import numpy as np
from scipy.signal import decimate
import os
from PIL import Image

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

def generate_dataset_unsupervised(data_path,mask_path,hsv=False):

    bear_img_path = data_path + "/bear/"
    bear_mask_path = mask_path+ "/bear/"

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
                    im = Image.open(di_img+"/"+path)
                    if hsv:
                        im = im.convert('HSV')
                    img = np.asarray(im)
                    #print(di_mask)
                    mask = np.asarray(Image.open(di_mask+"/"+path[:-3]+"png"))
                    temp1.append(img)
                    temp2.append(mask)

        imgs += temp1
        masks +=  temp2

    return imgs, masks
