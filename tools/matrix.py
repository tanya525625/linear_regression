# only for isinstance
import numpy as np


class Matrix:
    def __init__(self, array: list = None, n_rows: int = None, n_cols: int = None):
        if array is not None:
            # if it is vector
            if not (isinstance(array[0], list) or isinstance(array[0], np.ndarray)):
                self.n_cols = 1
            else:
                self.n_cols = len(array[0])

            self.arr = array
            self.n_rows = len(array)
            self.els_count = self.n_rows * self.n_cols

        if n_rows and n_cols:
            self.els_count = n_rows * n_cols
            self.arr = _make_zero_array(n_rows, n_cols)
            self.n_rows = n_rows
            self.n_cols = n_cols

    def sum_all_els(self):
        total = 0
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
        new_arr = Matrix(n_rows=self.n_cols, n_cols=self.n_rows);
        for i, row in enumerate(self.arr):
            for j, el in enumerate(self.arr[i]):
                new_arr.arr[j][i] = el

        return new_arr


def _make_zero_array(n_rows, n_cols):
    arr = []
    zero = 0

    for i in range(n_rows):
        arr.append([])
        for j in range(n_cols):
            arr[i].append(zero)

    return arr



# a = 3
# b = 5
# r = 0  # Чтобы было, чем заполнять
# mas = []
# for i in range(a):
#     mas.append([])
#     for j in range(b):
#         mas[i].append(r)
#         r += 1  # Чтобы заполнялось не одно и тоже
#
# arr = Matrix(mas)
# print(arr.arr)
# print(arr.sum_all_els())
# vect = [0, 1, 2, 3]
# vect = Matrix(vect)
# print(vect.sum_all_els())
# print(vect.n_cols)
# print(vect.find_mean_in_vect())

