from tools.matrix import Matrix
from tools.functions import multiply_vectors


class LinearRegression():
    @staticmethod
    def fit(X: Matrix, y: Matrix):
        pass

    @staticmethod
    def predict():
        pass


def _find_coeff(X: Matrix, y: Matrix):
    # weight matrix
    b = Matrix(n_rows=X.n_rows + 1, n_cols=1)
    mean_y = y.find_mean_in_vect()
    transposed_X = X.transpose()
    for i, row in enumerate(X.arr):
        row = Matrix(row)
        mean_x = row.find_mean_in_vect()
        ss_xy = _find_cross_deflection(row, y)
        ss_xx = _find_cross_deflection(row, row)
        b.arr[i+1] = ss_xy / ss_xx

    print(b.arr)


def _find_cross_deflection(vect1: Matrix, vect2: Matrix):
    mean_vect1 = vect1.find_mean_in_vect()
    mean_vect2 = vect2.find_mean_in_vect()
    n = vect1.n_rows
    composition = multiply_vectors(vect1, vect2)

    return composition.sum_all_els() - n * mean_vect1 * mean_vect2


a = 3
b = 5
r = 0  # Чтобы было, чем заполнять
mas = []
for i in range(a):
    mas.append([])
    for j in range(b):
        mas[i].append(r)
        r += 1  # Чтобы заполнялось не одно и тоже

# arr = Matrix(mas)
# print(arr.arr)
# print(arr.sum_all_els())
# vect = [0, 1, 2]
# vect = Matrix(vect)
# b = _find_coeff(arr, vect)
# print(b)
# model = LinearRegression()

# print(vect.sum_all_els())
# print(vect.n_cols)
# print(vect.find_mean_in_vect())






