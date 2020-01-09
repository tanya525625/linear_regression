from tools.matrix import Matrix, multiply_matrices, multiply_matrix_and_vector
from tools.functions import paste_ones_in_the_beginning


class LinearRegression:
    def __init__(self):
        # weight matrix
        self.b = None

    def fit(self, X: Matrix, y: Matrix):
        X = paste_ones_in_the_beginning(X)
        self.b = _find_coeff(X, y)

        return self

    def predict(self, X: Matrix):
        X = paste_ones_in_the_beginning(X)
        y_pred = multiply_matrix_and_vector(X, self.b)

        return y_pred.arr


def _find_coeff(X: Matrix, y: Matrix, ):
    # weight matrix

    transposed_X = X.transpose()
    composition = multiply_matrices(transposed_X, X)
    inversed_comp = composition.inverse()
    b = multiply_matrices(inversed_comp, transposed_X)
    b = multiply_matrix_and_vector(b, y)

    return b


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






