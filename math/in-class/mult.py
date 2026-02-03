


def matrix_multiplication(m1, m2):
    # verify if the matrices can be multiplied (columns of m1 == rows of m2)
    if len(m1[0]) != len(m2):
        return ("The multiplication is impossible")

    # multiply: for each row i in m1 and column j in m2, compute dot product
    return [[sum(m1[i][k] * m2[k][j] for k in range(len(m2)))
             for j in range(len(m2[0]))]
            for i in range(len(m1))]

solution = matrix_multiplication([[1, 3], [4, 6]], [[7, 8], [9, 10],[11, 12]])
print(solution)