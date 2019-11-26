from os.path import join
from PIL import Image
import numpy as np

def get_img_and_mask(path, name, id):
    full_img_path = join(path, "JPEGImages", "480p", name, ("00000" + str(id))[-5:] + ".jpg")
    img = Image.open(full_img_path)
    full_msk_path = join(path, "Annotations", "480p", name, ("00000" + str(id))[-5:] + ".png")
    msk = Image.open(full_msk_path)
    return img, msk


def set_width(img, msk, height):
    newsize = (int(img.size[0] * height / img.size[1]), height)
    return img.resize(newsize), msk.resize(newsize)

def img_to_array(img, msk):
    return np.asarray(img), (np.asarray(msk)//255).astype("uint8")

def show_mask(msk):
    yy=np.asarray(msk).copy()
    if np.max(msk)==1:
        yy=yy*255
    return Image.fromarray(yy)

def jaccard_index(img1, msk):
    """ Calculates the Jaccard index (IoU) measure for 2 detection bounding
        boxes in a frame.
        box1, box2: 2-tuples of the form (x, y), representing coordinates of
            the top left corner of the box.
        imshape: A 2-tuple (H, W), the shape of the image in which the boxes
            are detected.
    """
    a,b = img_to_array(img1,msk)
    plus = a+b
    i = np.where(plus == 2, 1, 0)
    print(i)
    j = np.where(plus != 0, 1, 0)
    print(j)
    intersection = i.sum()

    union = j.sum()

    return intersection / union


def multi_label_jacard(labels,msk):
    all_labels=sorted(list(set(labels)))
    #for labels in labels:
    jacard_index=[]
    for label in all_labels:
        labels==label
        plus=(labels==label)*1+msk.reshape(-1)
        i = np.where(plus == 2, 1, 0)
        j = np.where(plus != 0, 1, 0)
        intersection = i.sum()
        union = j.sum()
        jacard_index.append(intersection / union)
    return jacard_index
