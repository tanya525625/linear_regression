# only for isinstance
import numpy as np

from copy import deepcopy


class Matrix:
    def __init__(self, array: list = None, n_rows: int = None, n_cols: int = None):
        if array is not None:
            if isinstance(array, float):
                self.n_cols = 1
                self.n_rows = 1
            else:
                # if it is vector
                if not (isinstance(array[0], list) or isinstance(array[0], np.ndarray)):
                    self.n_cols = 1
                else:
                    self.n_cols = len(array[0])
                self.n_rows = len(array)

            self.arr = array
            self.els_count = self.n_rows * self.n_cols

        if n_rows and n_cols:
            self.els_count = n_rows * n_cols
            self.arr = _make_zero_array(n_rows, n_cols)
            self.n_rows = n_rows
            self.n_cols = n_cols

    def sum_all_els(self):
        total = 0
        if self.n_cols == 1 and self.n_rows == 1:
            return self.arr

        if self.n_cols != 1:
            for row in self.arr:
                for el in row:
                    total += el
        else:
            for el in self.arr:
                total += el

        return total

    def find_mean_in_vect(self):
        if self.n_cols != 1:
            raise TypeError("Vector hasn't dimension 1")

        return self.sum_all_els() / self.els_count

    def transpose(self):
        new_arr = Matrix(n_rows=self.n_cols, n_cols=self.n_rows)
        for i, row in enumerate(self.arr):
            for j, el in enumerate(self.arr[i]):
                new_arr.arr[j][i] = el

        return new_arr

    def determinant(self):
        det = 1
        triag_matr, sign = self.make_triangular()

        if self.n_rows != self.n_cols:
            raise ValueError('The matrix is not quadratic')
        for k in range(self.n_rows):
            det *= triag_matr.arr[k][k]

        return det * sign

    def make_triangular(self):
        sign = 1
        n = self.n_rows
        new_matrix = Matrix(n_rows=self.n_rows, n_cols=self.n_cols)
        new_matrix.arr = deepcopy(self.arr)
        arr = new_matrix.arr

        for k in range(n - 1):
            sign = _change_rows_due_to_max(new_matrix, k, sign)
            for i in range(k + 1, n):
                div = arr[i][k] / arr[k][k]
                for j in range(k, n):
                    arr[i][j] -= div * arr[k][j]

        new_matrix.arr = arr

        return new_matrix, sign

    def make_minor(self, i, j):
        minor = Matrix(n_rows=self.n_rows - 1, n_cols=self.n_cols - 1)
        arr = deepcopy(self.arr)
        minor.arr = [row[:j] + row[j + 1:] for row in (arr[:i] + arr[i + 1:])]

        return minor

    def inverse(self):
        det = self.determinant()
        if det == 0:
            raise ValueError('Determinant equals to zero')
        arr = deepcopy(self.arr)

        # special case for 2x2 matrix:
        if self.n_cols == 2:
            return Matrix([[arr[1][1] / det, -1 * arr[0][1] / det],
                    [-1 * arr[1][0] / det, arr[0][0] / det]])

        union_matrix = []
        for r in range(self.n_cols):
            union_matrix_row = []
            for c in range(self.n_cols):
                minor = self.make_minor(r, c)
                union_matrix_row.append(((-1) ** (r + c)) * minor.determinant())
            union_matrix.append(union_matrix_row)

        union_matrix = Matrix(union_matrix)
        union_matrix = union_matrix.transpose()
        union_matrix = mul_const_with_matrix(union_matrix, 1 / det)

        return union_matrix


def _make_zero_array(n_rows, n_cols):
    arr = []
    zero = 0
    if n_cols == 1:
        for i in range(n_rows):
            arr.append(zero)
    else:
        for i in range(n_rows):
            arr.append([])
            for j in range(n_cols):
                arr[i].append(zero)

    return arr


def _change_rows_due_to_max(array: Matrix, curr_k, sign):
    arr = array.arr

    max_element = arr[curr_k][curr_k]
    max_row = curr_k
    for i in range(curr_k + 1, array.n_rows):
        if abs(arr[i][curr_k]) > abs(max_element):
            max_element = arr[i][curr_k]
            max_row = i
    if max_row != curr_k:
        sign = sign * (-1)
        arr[curr_k], arr[max_row] = arr[max_row], arr[curr_k]

    return sign


def mul_const_with_vector(vector, const):
    if vector.n_cols != 1:
        raise TypeError("Vector hasn't dimension 1")

    res = Matrix(n_rows=vector.n_rows, n_cols=vector.n_cols)
    for i in range(vector.n_rows):
        if isinstance(res.arr, float):
            res.arr = vector.arr * const
        else:
            res.arr[i] = vector.arr[i] * const

    return res


def mul_const_with_matrix(matrix: Matrix, const):
    res = Matrix(n_rows=matrix.n_rows, n_cols=matrix.n_cols)
    for i in range(matrix.n_rows):
        for j in range(matrix.n_cols):
            res.arr[i][j] = matrix.arr[i][j] * const

    return res


def multiply_matrices(matr1: Matrix, matr2: Matrix):

    new_matrix = Matrix(n_rows=matr1.n_rows, n_cols=matr2.n_cols)
    if matr1.n_cols != matr2.n_rows:
        raise ValueError('Matrices are not consistent')
    for i in range(matr1.n_rows):
        for j in range(matr2.n_cols):
            for k in range(matr2.n_rows):
                #print(matr1.arr)
                #print(matr2.arr[k][j])
                new_matrix.arr[i][j] += matr1.arr[i][k] * matr2.arr[k][j]

    return new_matrix


def multiply_matrix_and_vector(matrix: Matrix, vector: Matrix):
    new_matrix = Matrix(n_rows=matrix.n_rows, n_cols=1)
    if matrix.n_cols != vector.n_rows:
        raise ValueError('Matrices are not consistent')
    for i in range(matrix.n_rows):
        for k in range(vector.n_rows):
            new_matrix.arr[i] += matrix.arr[i][k] * vector.arr[k]

    return new_matrix






a = 3
b = 3
r = 0  # Чтобы было, чем заполнять
mas = []
for i in range(a):
    mas.append([])
    for j in range(b):
        mas[i].append(r)
        r += 1  # Чтобы заполнялось не одно и тоже


# #mas = solve_gauss(mas)
# mas = Matrix(mas)
# #new_matrix = mas.make_triangular()
# # new_matr = mas.transpose()
# # print(new_matr.arr)
#
#
# mas = [[1, 1, 19], [4, 4, 6], [6, 13, 8]]
# mas = Matrix(mas)
#
#
# print((mas.inverse()).arr)
#
# arr = Matrix(mas)
# print(arr.arr)
# print(arr.sum_all_els())
# vect = [0, 1, 2, 3]
# vect = Matrix(vect)
# print(vect.sum_all_els())
# print(vect.n_cols)
# print(vect.find_mean_in_vect())


