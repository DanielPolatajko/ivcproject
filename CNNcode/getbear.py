import sys
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import data_provider as data_providers
from experiment_builder import ExperimentBuilder
from models import ShallowNetwork, DeeperNetwork, DeepestNetwork
import storage_utils as storage_utils
from utils import generate_dataset_static, generate_dataset_temporal, jaccard_loss, down_sample

from PIL import Image
from scipy.signal import decimate

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import tqdm
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import numpy as np
import os

def overlay_segment(img, segment):
    back = img.convert("RGBA")
    fore = segment.convert("RGBA")

    temp = list(fore.getdata())
    for i, pixel in enumerate(temp):
        if pixel[:3] == (255,255,255):
            temp[i] = (0,0,255,100)
        else:
            temp[i] = (255,0,0,0)

    fore.putdata(temp)

    back.paste(fore, (0,0),fore)

    return back

def main():

    new_model = DeepestNetwork((25,3,120,214))

    N=4

    cwd = Path(os.getcwd())
    par = cwd.parent
    data_path = str(par / "data/DAVIS//JPEGImages/480p/")
    mask_path = str(par / "data/DAVIS/Annotations/480p/")

    tvt_split = (0.5,0.7)

    X_train_t, X_val_t, X_test_t, y_train_t, y_val_t, y_test_t = generate_dataset_temporal(data_path, mask_path,tvt_split, N)

    X_train_t = np.array(X_train_t).swapaxes(-1,-3).swapaxes(-2,-1)
    X_val_t = np.array(X_val_t).swapaxes(-1,-3).swapaxes(-2,-1)
    X_test_t = np.array(X_test_t).swapaxes(-1,-3).swapaxes(-2,-1)
    print(X_train_t.shape)
    print(X_val_t.shape)
    print(X_test_t.shape)
    y_train_t = np.array(y_train_t)
    y_val_t = np.array(y_val_t)
    y_test_t = np.array(y_test_t)
    print(y_train_t.shape)
    print(y_val_t.shape)
    print(y_test_t.shape)

    batch_size = 25
    train_data_t = data_providers.DataProvider(X_train_t,y_train_t,batch_size,shuffle_order=True)
    val_data_t = data_providers.DataProvider(X_val_t,y_val_t,batch_size,shuffle_order=True)
    test_data_t = data_providers.DataProvider(X_test_t,y_test_t,batch_size,shuffle_order=True)

    eb = ExperimentBuilder(new_model, "get_bear", 1, train_data_t, val_data_t, test_data_t, True)

    model_path = Path(os.getcwd())
    model_path = model_path / "static_run_deepest" / "saved_models"

    bear_path = Path(os.getcwd()).parent / "data" / "DAVIS" / "JPEGImages" / "480p" / "bear"

    bear = np.asarray(Image.open(str(bear_path/"00001.jpg")).convert(mode="RGB"))

    inp = torch.Tensor(down_sample(np.asarray(bear),4).swapaxes(0,2).swapaxes(1,2)).unsqueeze(0)

    out = eb.get_bear(model_path, inp)
    out = out.squeeze()

    predicted = F.sigmoid(out) > 0.5

    mask = predicted.cpu().numpy().astype('uint8')

    mask = 255 * mask

    mask_img = Image.fromarray(mask, mode='L')

    bear = down_sample(bear,4)
    bear = Image.fromarray(bear)


    overlay = overlay_segment(bear, mask_img)

    overlay.save("cnnbear.png")


if __name__ == "__main__":
    main()
