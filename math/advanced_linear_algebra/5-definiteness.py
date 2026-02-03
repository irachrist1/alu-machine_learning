#!/usr/bin/env python3
"""Module for determining the definiteness of a matrix."""
import numpy as np


def definiteness(matrix):
    """
    Calculate the definiteness of a matrix.

    Args:
        matrix: A numpy.ndarray of shape (n, n) whose definiteness
                should be calculated

    Returns:
        String indicating the definiteness: 'Positive definite',
        'Positive semi-definite', 'Negative semi-definite',
        'Negative definite', 'Indefinite', or None

    Raises:
        TypeError: If matrix is not a numpy.ndarray
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    
    # Check if matrix is valid (2D and square)
    if len(matrix.shape) != 2:
        return None
    
    if matrix.shape[0] != matrix.shape[1]:
        return None
    
    # Check if matrix is empty
    if matrix.shape[0] == 0:
        return None
    
    # Check if matrix is symmetric (required for definiteness)
    if not np.allclose(matrix, matrix.T):
        return None
    
    # Calculate eigenvalues
    try:
        eigenvalues = np.linalg.eigvals(matrix)
    except Exception:
        return None
    
    # Check for numerical errors
    if not np.all(np.isfinite(eigenvalues)):
        return None
    
    # Determine definiteness based on eigenvalues
    positive = np.all(eigenvalues > 0)
    negative = np.all(eigenvalues < 0)
    non_negative = np.all(eigenvalues >= 0)
    non_positive = np.all(eigenvalues <= 0)
    
    # Use a small tolerance for zero comparison
    tolerance = 1e-10
    has_zero = np.any(np.abs(eigenvalues) < tolerance)
    
    if positive:
        return "Positive definite"
    elif non_negative and has_zero:
        return "Positive semi-definite"
    elif negative:
        return "Negative definite"
    elif non_positive and has_zero:
        return "Negative semi-definite"
    else:
        return "Indefinite"
