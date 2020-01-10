from tools.matrix import Matrix, multiply_matrices, multiply_matrix_and_vector, paste_ones_in_the_beginning


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


