from tools.matrix import Matrix


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


# vect1 = [0, 1, 2, 3]
# vect1 = Matrix(vect1)
# vect2 = [0, 2, 4, 3]
# vect2 = Matrix(vect2)
# print(multiply_vectors(vect1, vect2).arr)
