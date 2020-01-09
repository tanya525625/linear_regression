from tools.matrix import Matrix, multiply_matrices, multiply_matrix_and_vector
from tools.functions import paste_ones_in_the_beginning, \
    make_identity_matrix, sum_matrices, mul_const_with_matrix


class RidgeRegularisation:
    def __init__(self, alpha=1):
        # weight matrix
        self.b = None
        self.alpha = alpha

    def fit(self, X: Matrix, y: Matrix):
        X = paste_ones_in_the_beginning(X)
        self.b = _find_coeff(X, y, self.alpha)

        return self

    def predict(self, X: Matrix):
        X = paste_ones_in_the_beginning(X)
        y_pred = multiply_matrix_and_vector(X, self.b)

        return y_pred.arr


def _find_coeff(X: Matrix, y: Matrix, alpha):
    # inv(X'X + alpha*I)*X'y

    transposed_X = X.transpose()
    identity_matrix = make_identity_matrix(X.n_cols)
    composition = multiply_matrices(transposed_X, X)
    comp_aplha_ident_matr = mul_const_with_matrix(identity_matrix, alpha)
    matrices_sum = sum_matrices(comp_aplha_ident_matr, composition)
    inversed_comp = matrices_sum.inverse()
    b = multiply_matrices(inversed_comp, transposed_X)
    b = multiply_matrix_and_vector(b, y)

    return b