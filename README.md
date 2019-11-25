# Investigating techniques for image and video segmentation

## Image and Vision Computing Assignment 2019-20

### Daniel Polatajko and Wim Zwart

Included below is a description of how to repeat the experiments that informed our report.

### Conda environment

We used Anaconda3, with two separate environments for the classical methods and for deep learning.

For the deep learning environment, navigate to the folder CNNcode and run:

`conda env create -f environment.yml`


### Running Experiments

For the $K$-means experiments, navigate to the classicalcode directory and run:

`python kmeans.py`

The results will be printed to stdout as the experiment runs.



For the deep learning experiments, navigate to the CNNcode directory, and run:

`python main.py`

The results of the experiments (cross entropy loss, accuracy and jaccard score metrics for training, validation and test set) will be output to csv files in a directory called result_outputs, each within a directory for the experiment being run (there are 6). There will also be a directory called saved_models, with saved versions of each model after every epoch, in each of the experiment output directories.
