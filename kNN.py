
from sklearn.base import BaseEstimator,ClassifierMixin
from scipy.spatial.distance import cdist
import numpy as np

class kNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors:int = 3):
        self.n_neighbors = n_neighbors
    def fit(self, X, y):
        self.x_train = X.copy()
        self.y_train = y.copy()
        return self

    def predict(self, X):
        dist = cdist(X, self.x_train, metric='euclidean')
        top = np.argpartition(dist,self.n_neighbors, axis=1)[:, :self.n_neighbors]
        neighbor_labels = self.y_train[top]
        predictions = np.sign(np.sum(neighbor_labels, axis=1)) # TODO: what to do when sum == 0?

        return predictions
