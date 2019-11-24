""" Utility functions for training deep CNNs, loading datasets and evaluating
    IoU score (Jaccard index)"""

import os

import torch

import numpy as np

from scipy.signal import decimate

from PIL import Image




epsilon = 1e-10
def jaccard_loss(x,y):

    x = x.float()
    x = torch.clamp(x,0,1).float()
    y = torch.clamp(y,0,1).float()
    #print(y.numpy()[0][y.numpy()[0]>0])

    plus = x.add(y)
    #print(plus)
    i = torch.eq(plus,2).float()
    #print(i)
    temp = torch.clamp(plus,0,1)
    j = torch.eq(temp,1).float()
    #print(j)
    intersection = torch.sum(i,(-1,-2))

    union = torch.sum(j,(-1,-2))

    #print(intersection)
    #print(union)

    return ((intersection+epsilon) / (epsilon+union)).mean()

def down_sample(img,n):
    img_d=decimate(img, n, n=2, ftype='fir',axis=0, zero_phase=True)
    img_d=decimate(img_d, n, n=2, ftype='iir',axis=1, zero_phase=True)
    return img_d.astype("uint8")

def generate_pathlists(data_path, mask_path):

    vids_img = []
    for dir,_,_ in os.walk(data_path):
        vids_img.append(dir)

    vids_mask = []
    for dir,_,_ in os.walk(mask_path):
        vids_mask.append(dir)

    return vids_img, vids_mask

def generate_dataset_temporal(data_path, mask_path,tvt_split=(0.5,0.7), down_sample_factor=1):

    print("Generating temporally linked dataset...")

    # define path to bear folder in order to deal with that one degenerate image
    bear_img_path = data_path + "/bear/"
    bear_mask_path = mask_path+ "/bear/"

    vids_img, vids_mask = generate_pathlists(data_path,mask_path)

    X_train_t = []
    X_val_t = []
    X_test_t = []
    y_train_t = []
    y_val_t = []
    y_test_t = []

    for i in range(len(vids_img[1:])):
        di_img = vids_img[i+1]
        di_mask = vids_mask[i+1]
        temp1 = []
        temp2 = []
        ix1 = []
        ix2 = []
        for _,_,paths in os.walk(di_img):
            for path in paths:
                if di_img+"/"+path == bear_img_path + "00077.jpg":
                    pass
                else:
                    temp1.append(di_img+"/"+path)
                    temp2.append(di_mask+"/"+path[:-3]+"png")
                    ix1.append(int(path[:5]))
                    ix2.append(int(path[:5]))

        sortdatapaths = [i for _,i in sorted(zip(ix1,temp1))]
        sortmaskpaths = [i for _,i in sorted(zip(ix2,temp2))]

        temp1 = []
        temp2 = []

        if down_sample_factor > 1:
            sortdata = []
            sortmask = []
            for i in sortdatapaths:
                im = Image.open(i)
                sortdata.append(down_sample(np.asarray(im),down_sample_factor))
                im.close()
            for i in sortmaskpaths:
                im = Image.open(i)
                sortmask.append(down_sample(np.asarray(im),down_sample_factor))
                im.close()
        else:
            sortdata = []
            sortmask = []
            for i in sortdatapaths:
                im = Image.open(i)
                sortdata.append(np.asarray(im))
                im.close()
            for i in sortmaskpaths:
                im = Image.open(i)
                sortmask.append(np.asarray(im))
                im.close()


        l = len(sortdata)
        tr,v = tvt_split

        temp_train = sortdata[:int(tr*l)]
        temp_val = sortdata[int(tr*l):int(v*l)]
        temp_test = sortdata[int(v*l):]

        sortdata = []

        X_train_t += [np.concatenate((temp_train[i],temp_train[i+1]), axis=2) for i in range(len(temp_train)-1)]
        X_val_t += [np.concatenate((temp_val[i],temp_val[i+1]), axis=2) for i in range(len(temp_val)-1)]
        X_test_t += [np.concatenate((temp_test[i],temp_test[i+1]), axis=2) for i in range(len(temp_test)-1)]

        #made some changes
        y_train_t += sortmask[1:int(tr*l)]
        y_val_t += sortmask[int(tr*l)+1:int(v*l)]
        y_test_t += sortmask[int(v*l)+1:]

    print("Generating dataset complete!")

    return X_train_t, X_val_t, X_test_t, y_train_t, y_val_t, y_test_t

def generate_dataset_static(data_path, mask_path,tvt_split=(0.5,0.7), down_sample_factor=1):

    print("Generating dataset...")

    bear_img_path = data_path + "/bear/"
    bear_mask_path = mask_path+ "/bear/"

    vids_img, vids_mask = generate_pathlists(data_path,mask_path)


    X_train = []
    X_val = []
    X_test = []
    y_train = []
    y_val = []
    y_test = []


    for i in range(len(vids_img[1:])):
        di_img = vids_img[i+1]
        di_mask = vids_mask[i+1]
        temp1 = []
        temp2 = []
        for _,_,paths in os.walk(di_img):
            for path in paths:
                if di_img+"/"+path == bear_img_path + "00077.jpg":
                    pass
                else:
                    img = np.asarray(Image.open(di_img+"/"+path))
                    mask = np.asarray(Image.open(di_mask+"/"+path[:-3]+"png"))
                    if down_sample_factor > 1:
                        img = down_sample(img, down_sample_factor)
                        mask = down_sample(mask, down_sample_factor)
                    temp1.append(img)
                    temp2.append(mask)
        l = len(temp1)
        tr,v = tvt_split
        X_train += temp1[:int(tr*l)]
        X_val += temp1[int(tr*l):int(v*l)]
        X_test += temp1[int(v*l):]
        y_train += temp2[:int(tr*l)]
        y_val += temp2[int(tr*l):int(v*l)]
        y_test += temp2[int(v*l):]

    print("Generating dataset complete!")

    return X_train, X_val, X_test, y_train, y_val, y_test
