import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import Normalizer, StandardScaler

def _data_transformation(X):
    normalizer = Normalizer()
    #normalizer.fit(X)
    #return normalizer.transform(X)
    
    scaler = StandardScaler()
    scaler.fit(X)
    normalizer.fit(scaler.transform(X))
    return normalizer.transform(scaler.transform(X))

def load_datasets():
    # import data
    df1 = pd.read_fwf('datasets/mfeat-fac', header=None) # mfeat-fac: 216 profile correlations;
    df2 = pd.read_fwf('datasets/mfeat-kar', header=None) # mfeat-kar: 64 Karhunen-Love coefficients;
    df3 = pd.read_fwf('datasets/mfeat-fou', header=None) # mfeat-fou: 76 Fourier coefficients of the character shapes;

    # select features and target data
    y = np.array([j for j in range(10) for i in range(200)]) # make target for classification.

    return [_data_transformation(df1.values), _data_transformation(df2.values), _data_transformation(df3.values)], y

def dissimilarity_matrices(X):
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html#sklearn.metrics.pairwise.euclidean_distances
    return np.array([euclidean_distances(x) for x in X])
