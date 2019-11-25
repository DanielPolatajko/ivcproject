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
from utils import generate_dataset_static, generate_dataset_temporal, jaccard_loss

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

    model_path = Path(os.getcwd())
    model_path = model_path / "static_run_deepest" / "saved_models"

    new_model.load_state_dict(torch.load(model_path))

    bear_path = Path(os.getcwd()).parent / "data" / "JPEGImages" / "480p" / "bear"

    bear = Image.open(bear_path+"00001.jpg").convert(mode="RGB")

    inp = down_sample(np.asarray(bear),4).swapaxes(1,3).swapaxes(2,3)

    out = new_model.forward(inp)
    out = out.squeeze()

    predicted = F.sigmoid(out) > 0.5

    mask = predicted.numpy().astype('uint8')

    mask = 255 * mask

    mask_img = Image.fromarray(mask, mode='L')

    overlay = overlay_segment(bear, mask_img)

    overlay.save("cnnbear.png")


if __name__ == "__main__":
    main()
