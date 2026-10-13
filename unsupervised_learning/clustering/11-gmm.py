#!/usr/bin/env python3
"""Fit a full-covariance Gaussian mixture using scikit-learn."""
import sklearn.mixture


def gmm(X, k):
    """Return priors, means, covariances, point labels, and model BIC."""
    model = sklearn.mixture.GaussianMixture(n_components=k)
    clss = model.fit_predict(X)
    return (model.weights_, model.means_, model.covariances_, clss,
            model.bic(X))
