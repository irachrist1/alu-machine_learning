#!/usr/bin/env python3
"""Run K-means using scikit-learn."""
import sklearn.cluster


def kmeans(X, k):
    """Return learned centroids and labels using k-means++ initialization."""
    model = sklearn.cluster.KMeans(n_clusters=k, n_init=10)
    clss = model.fit_predict(X)
    return model.cluster_centers_, clss
