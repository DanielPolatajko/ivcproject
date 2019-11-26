from pystruct import learners
import pystruct.models as crfs
from pystruct.utils import SaveLogger
import numpy as np
from PIL import Image

from pystruct.learners import OneSlackSSVM
from pystruct.utils import make_grid_edges, edge_list_to_features
from pystruct.models import EdgeFeatureGraphCRF


def neighborhood_feature(x):
    """Add a 3x3 neighborhood around each pixel as a feature."""
    # we could also use a four neighborhood, that would work even better
    # but one might argue then we are using domain knowledge ;)
    features = np.zeros((x.shape[0], x.shape[1], 3, 9))
    # position 3 is background.
    features[1:, 1:, :, 0] = x[:-1, :-1, :]
    features[:, 1:, :, 1] = x[:, :-1, :]
    features[:-1, 1:, :, 2] = x[1:, :-1, :]
    features[1:, :, :, 3] = x[:-1, :, :]
    features[:-1, :-1, :, 4] = x[1:, 1:, :]
    features[:-1, :, :, 5] = x[1:, :, :]
    features[1:, :-1, :, 6] = x[:-1, 1:, :]
    features[:, :-1, :, 7] = x[:, 1:, :]
    features[:, :, :, 8] = x[:, :, :]
    return features.reshape(x.shape[0] * x.shape[1], -1)

def make_directions(X):
    edges = make_grid_edges(X)
    right, down = make_grid_edges(X, return_lists=True)
    edges = np.vstack([right, down])
    edge_features_directions = edge_list_to_features([right, down])
    features = neighborhood_feature(X)
    return [(features, edges, edge_features_directions)]

def predict (model, X):
    X_val_directions = make_directions(np.asarray(X))
    Y_val = model.predict(X_val_directions)[0].reshape((np.asarray(X).shape[0],np.asarray(X).shape[1]))
    Y_val = np.asarray(Y_val).astype("float").astype("uint8")
    return (Y_val)

def train(X,y):
    X_train_directions = make_directions(X)
    Y_train_flat = [y.ravel()]
    inference = 'qpbo'
    # first, train on X with directions only:
    crf = EdgeFeatureGraphCRF(inference_method=inference)
    ssvm = OneSlackSSVM(crf, inference_cache=50, C=.1, tol=.1, max_iter=10,
                        n_jobs=1,show_loss_every=1)
    ssvm.fit(X_train_directions, Y_train_flat, warm_start=False)
    return ssvm