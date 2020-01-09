from tools.matrix import Matrix

from copy import deepcopy


def make_identity_matrix(dim):
    identity_matr = Matrix(n_cols=dim, n_rows=dim)
    for i in range(dim):
        identity_matr.arr[i][i] = 1

    return identity_matr


def sum_matrices(matr1: Matrix, matr2: Matrix):
    if matr1.n_rows != matr2.n_rows or matr1.n_cols != matr2.n_cols:
        raise ValueError("Dimensions are not equal")

    new_matrix = Matrix(n_rows=matr1.n_rows, n_cols=matr2.n_cols)

    for i in range(matr1.n_rows):
        for j in range(matr2.n_cols):
            new_matrix.arr[i][j] = matr1.arr[i][j] + matr2.arr[i][j]

    return new_matrix


def multiply_vectors(vect1: Matrix, vect2: Matrix):
    if vect1.n_cols != 1 or vect2.n_cols != 1:
        raise TypeError("Vector hasn't dimension 1")

    if vect1.n_rows != vect2.n_rows:
        raise ValueError("Dimensions are not equal")

    composition = Matrix(n_rows=vect1.n_rows, n_cols=vect2.n_cols)
    for i in range(vect1.n_rows):
        composition.arr[i] = vect1.arr[i] * vect2.arr[i]

    return composition


def vectors_subtraction(vect1, vect2):
    if vect1.n_cols != 1 or vect2.n_cols != 1:
        raise TypeError("Vector hasn't dimension 1")

    if vect1.n_rows != vect2.n_rows:
        raise ValueError("Dimensions are not equal")

    subtraction = Matrix(n_rows=vect1.n_rows, n_cols=vect1.n_cols)
    for i in range(vect1.n_rows):
        subtraction.arr[i] = vect1.arr[i] - vect2.arr[i]

    return subtraction


def sum_const_with_vector(vector: Matrix, const):
    if vector.n_cols != 1:
        raise TypeError("Vector hasn't dimension 1")

    res = Matrix(n_rows=vector.n_rows, n_cols=vector.n_cols)
    for i in range(vector.n_rows):
        res.arr[i] = vector.arr[i] + const

    return res


def mul_const_with_matrix(matrix: Matrix, const):
    res = Matrix(n_rows=matrix.n_rows, n_cols=matrix.n_cols)
    for i in range(matrix.n_rows):
        for j in range(matrix.n_cols):
            res.arr[i][j] = matrix.arr[i][j] * const

    return res


def ones(size):
    new_list = []
    for i in range(size):
        new_list.append(1)

    return new_list


def paste_ones_in_the_beginning(matrix: Matrix):
    arr = deepcopy(matrix.arr)

    new_arr = []
    for i, row in enumerate(arr):
        if isinstance(row, float):
            new_arr.append([1] + [row])
        else:
            new_arr.append([1] + row.tolist())

    return Matrix(new_arr)


def rmse(y_pred, y_true):
    rmse = 0
    n =  len(y_pred)

    for i in range(n):
        rmse += (y_true[i] - y_pred[i]) ** 2

    return (rmse / n) ** (1 / 2)




# vect1 = [0, 1, 2, 3]
# vect1 = Matrix(vect1)
# vect2 = [0, 2, 4, 3]
# vect2 = Matrix(vect2)
# print(multiply_vectors(vect1, vect2).arr)
