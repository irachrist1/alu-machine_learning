#!/usr/bin/env python3
"""Build and display hierarchical clusters with Ward linkage."""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Display a colored dendrogram and return distance-cut cluster labels."""
    linkage = scipy.cluster.hierarchy.linkage(X, method='ward')
    scipy.cluster.hierarchy.dendrogram(linkage, color_threshold=dist)
    plt.show()
    return scipy.cluster.hierarchy.fcluster(
        linkage, dist, criterion='distance')
