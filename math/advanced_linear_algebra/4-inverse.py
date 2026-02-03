#!/usr/bin/env python3
"""Module for calculating the inverse of a matrix."""


def determinant(matrix):
    """
    Calculate the determinant of a matrix.

    Args:
        matrix: A list of lists representing the matrix

    Returns:
        The determinant of the matrix
    """
    # Base case: 1x1 matrix
    if len(matrix) == 1:
        return matrix[0][0]
    
    # Base case: 2x2 matrix
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    # Recursive case: use cofactor expansion along first row
    det = 0
    for j in range(len(matrix)):
        # Create submatrix by removing first row and j-th column
        submatrix = []
        for i in range(1, len(matrix)):
            row = []
            for k in range(len(matrix)):
                if k != j:
                    row.append(matrix[i][k])
            submatrix.append(row)
        
        # Calculate cofactor and add to determinant
        cofactor = ((-1) ** j) * matrix[0][j] * determinant(submatrix)
        det += cofactor
    
    return det


def minor(matrix):
    """
    Calculate the minor matrix of a matrix.

    Args:
        matrix: A list of lists whose minor matrix should be calculated

    Returns:
        The minor matrix of matrix
    """
    n = len(matrix)
    
    # Special case: 1x1 matrix
    if n == 1:
        return [[1]]
    
    # Calculate minor matrix
    minor_matrix = []
    for i in range(n):
        minor_row = []
        for j in range(n):
            # Create submatrix by removing i-th row and j-th column
            submatrix = []
            for row_idx in range(n):
                if row_idx != i:
                    row = []
                    for col_idx in range(n):
                        if col_idx != j:
                            row.append(matrix[row_idx][col_idx])
                    submatrix.append(row)
            
            # Calculate determinant of submatrix
            minor_row.append(determinant(submatrix))
        minor_matrix.append(minor_row)
    
    return minor_matrix


def cofactor(matrix):
    """
    Calculate the cofactor matrix of a matrix.

    Args:
        matrix: A list of lists whose cofactor matrix should be calculated

    Returns:
        The cofactor matrix of matrix
    """
    n = len(matrix)
    
    # Get minor matrix
    minor_matrix = minor(matrix)
    
    # Apply checkerboard pattern of signs
    cofactor_matrix = []
    for i in range(n):
        cofactor_row = []
        for j in range(n):
            sign = ((-1) ** (i + j))
            cofactor_row.append(sign * minor_matrix[i][j])
        cofactor_matrix.append(cofactor_row)
    
    return cofactor_matrix


def adjugate(matrix):
    """
    Calculate the adjugate matrix of a matrix.

    Args:
        matrix: A list of lists whose adjugate matrix should be calculated

    Returns:
        The adjugate matrix of matrix
    """
    n = len(matrix)
    
    # Get cofactor matrix
    cofactor_matrix = cofactor(matrix)
    
    # Transpose the cofactor matrix to get adjugate
    adjugate_matrix = []
    for j in range(n):
        adjugate_row = []
        for i in range(n):
            adjugate_row.append(cofactor_matrix[i][j])
        adjugate_matrix.append(adjugate_row)
    
    return adjugate_matrix


def inverse(matrix):
    """
    Calculate the inverse of a matrix.

    Args:
        matrix: A list of lists whose inverse should be calculated

    Returns:
        The inverse of matrix, or None if matrix is singular

    Raises:
        TypeError: If matrix is not a list of lists
        ValueError: If matrix is not square or is empty
    """
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    
    if len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    
    # Check if it's a list of lists
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    
    # Check if matrix is empty
    if len(matrix) == 1 and len(matrix[0]) == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    
    # Check if matrix is square
    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    
    # Calculate determinant
    det = determinant(matrix)
    
    # If determinant is 0, matrix is singular
    if det == 0:
        return None
    
    # Get adjugate matrix
    adj = adjugate(matrix)
    
    # Divide adjugate by determinant
    inverse_matrix = []
    for i in range(n):
        inverse_row = []
        for j in range(n):
            inverse_row.append(adj[i][j] / det)
        inverse_matrix.append(inverse_row)
    
    return inverse_matrix
