from tools.matrix import Matrix
from tools.functions import multiply_vectors, mul_const_with_vector, sum_const_with_vector


class LinearRegression():
    def __init__(self):
        # weight matrix
        self.b = None

    def fit(self, X: Matrix, y: Matrix):
        self.b = _find_coeff(X, y)
        return self

    def predict(self, X: Matrix):
        y_pred = Matrix(n_rows=X.n_rows, n_cols=1)
        if isinstance(X.arr, float):
            composition_vector = mul_const_with_vector(X, self.b.arr[1])
            y_pred.arr = composition_vector.sum_all_els()
        else:
            for i, X_row in enumerate(X.arr):
                composition_vector = mul_const_with_vector(Matrix(X_row), self.b.arr[i+1])
                y_pred.arr[i] = composition_vector.sum_all_els()

        y_pred = sum_const_with_vector(y_pred, self.b.arr[0])
        return y_pred.arr


def _find_coeff(X: Matrix, y: Matrix):
    # weight matrix
    b = Matrix(n_rows=X.n_rows + 1, n_cols=1)

    # for i, row in enumerate(X.arr):
    #     row = Matrix(row)
        #print(row.arr)
        # if isinstance(row.arr, float):
        #     x_mean = X.find_mean_in_vect()
        #     y_mean = y.find_mean_in_vect()
        #     ss_xy = (row.arr - x_mean) * (y.arr - y_mean)
        #     ss_xx = (row.arr - x_mean) * (row.arr - x_mean)
        # else:
        #     ss_xy = _find_cross_deflection(row, y)
        #     ss_xx = _find_cross_deflection(row, row)

    #     b.arr[i+1] = ss_xy / ss_xx
    #
    # b.arr[0] = _find_b0(X, y, b)
    return b


def _find_cross_deflection(vect1: Matrix, vect2: Matrix):
    mean_vect1 = vect1.find_mean_in_vect()
    mean_vect2 = vect2.find_mean_in_vect()
    res = 0
    for i in range(vect1.n_rows):
        res += (vect1.arr[i] - mean_vect1) * (vect2.arr[i] - mean_vect2)

    return res


def _find_b0(X: Matrix, y: Matrix, b: Matrix):
    b0 = y.find_mean_in_vect()
    for i, X_row in enumerate(X.arr):
        b0 -= b.arr[i+1] * Matrix(X_row).find_mean_in_vect()

    return b0


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






