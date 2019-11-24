import sys
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import data_provider as data_providers
from experiment_builder import ExperimentBuilder
from models import ShallowNetwork, DeeperNetwork
import storage_utils as storage_utils
from utils import generate_dataset_static, generate_dataset_temporal, jaccard_loss

from PIL import Image
#from matplotlib.pyplot import imshow
#from scipy.spatial.distance import cdist
from scipy.signal import decimate

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import tqdm
import time

def main():

    N = 1

    # define paths to image and mask files
    cwd = Path(os.getcwd())
    par = cwd.parent
    data_path = str(par / "data/DAVIS//JPEGImages/480p/")
    mask_path = str(par / "data/DAVIS/Annotations/480p/")

    X_train, X_val, X_test, y_train, y_val, y_test = generate_dataset_static(data_path,mask_path, N)
    X_train_t, X_val_t, X_test_t, y_train_t, y_val_t, y_test_t = generate_dataset_temporal(data_path, mask_path, N)

    X_train = np.array(X_train).swapaxes(1,3).swapaxes(2,3)
    X_val = np.array(X_val).swapaxes(1,3).swapaxes(2,3)
    X_test = np.array(X_test).swapaxes(1,3).swapaxes(2,3)
    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)
    print(y_train.shape)
    print(y_val.shape)
    print(y_test.shape)

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

    batch_size=25
    train_data = data_providers.DataProvider(X_train,y_train,batch_size,shuffle_order=True)
    val_data = data_providers.DataProvider(X_val,y_val,batch_size,shuffle_order=True)
    test_data = data_providers.DataProvider(X_test,y_test,batch_size,shuffle_order=True)

    train_data_t = data_providers.DataProvider(X_train_t,y_train_t,batch_size,shuffle_order=True)
    val_data_t = data_providers.DataProvider(X_val_t,y_val_t,batch_size,shuffle_order=True)
    test_data_t = data_providers.DataProvider(X_test_t,y_test_t,batch_size,shuffle_order=True)

    inputs_shape = X_train[:batch_size].shape
    inputs_shape

    inputs_shape_t = X_train_t[:batch_size].shape
    inputs_shape_t

    static_net_shallow=ShallowNetwork(input_shape=inputs_shape)
    static_net_deeper=ShallowNetwork(input_shape=inputs_shape)
    temporal_net_shallow = ShallowNetwork(input_shape=inputs_shape_t)
    temporal_net_deeper = ShallowNetwork(input_shape=inputs_shape_t)

    experiment_name= "static_run_shallow"
    num_epochs = 30
    use_gpu=False
    continue_from_epoch=-1

    experiment_1 = ExperimentBuilder(network_model=static_net_shallow,
                                        experiment_name=experiment_name,
                                        num_epochs=num_epochs,
                                        use_gpu=use_gpu,
                                        continue_from_epoch=continue_from_epoch,
                                        train_data=train_data, val_data=val_data,
                                        test_data=test_data)  # build an experiment object
    experiment_metrics, test_metrics = experiment_1.run_experiment()  # run experiment and return experiment metrics

    experiment_name= "static_run_deeper"
    num_epochs = 30
    use_gpu=True
    continue_from_epoch=-1

    experiment_2 = ExperimentBuilder(network_model=static_net_shallow,
                                        experiment_name=experiment_name,
                                        num_epochs=num_epochs,
                                        use_gpu=use_gpu,
                                        continue_from_epoch=continue_from_epoch,
                                        train_data=train_data, val_data=val_data,
                                        test_data=test_data)  # build an experiment object
    experiment_metrics, test_metrics = experiment_1.run_experiment()  # run experiment and return experiment metrics

    experiment_name= "temporal_run_shallow"
    num_epochs = 30
    use_gpu=True
    continue_from_epoch=-1

    experiment_3 = ExperimentBuilder(network_model=temporal_net_shallow,
                                        experiment_name=experiment_name,
                                        num_epochs=num_epochs,
                                        use_gpu=use_gpu,
                                        continue_from_epoch=continue_from_epoch,
                                        train_data=train_data_t, val_data=val_data_t,
                                        test_data=test_data_t)  # build an experiment object
    experiment_metrics, test_metrics = experiment_1.run_experiment()  # run experiment and return experiment metrics

    experiment_name= "temporal_run_deeper"
    num_epochs = 30
    use_gpu=True
    continue_from_epoch=-1

    experiment_4 = ExperimentBuilder(network_model=temporal_net_deeper,
                                        experiment_name=experiment_name,
                                        num_epochs=num_epochs,
                                        use_gpu=use_gpu,
                                        continue_from_epoch=continue_from_epoch,
                                        train_data=train_data_t, val_data=val_data_t,
                                        test_data=test_data_t)  # build an experiment object
    experiment_metrics, test_metrics = experiment_1.run_experiment()  # run experiment and return experiment metrics

if __name__ == '__main__':
    main()
