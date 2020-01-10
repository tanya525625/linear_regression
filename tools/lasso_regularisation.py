

from tools.matrix import Matrix, multiply_matrix_and_vector, vectors_subtraction, \
    mul_vector_to_matrix, mul_const_with_vector, sum_vectors, paste_ones_in_the_beginning


class LassoRegularisation:
    def __init__(self, alpha=1, learning_rate=0.1, degree=10, iter_count=100):
        # weight matrix
        self.b = None
        self.iter_count = iter_count
        self.alpha = alpha
        self.degree = degree
        self.learning_rate = learning_rate

    def fit(self, X: Matrix, y: Matrix):
        self.b = Matrix(n_rows=X.n_cols + 1, n_cols=1)
        self.b.arr = [-1.0] * self.b.n_rows

        # X = paste_ones_in_the_beginning(X)
        X = _make_poly_list(X, self.degree)
        X = X.transpose()
        X = X.normalize_matrix()

        self.b = _find_coeff(X, y, self.b,
                             self.alpha, self.iter_count)
        return self

    def predict(self, X: Matrix):
        X = paste_ones_in_the_beginning(X)
        y_pred = multiply_matrix_and_vector(X, self.b)

        return y_pred.arr


def _make_poly_list(matr: Matrix, degree: int):
    matr = paste_ones_in_the_beginning(matr)
    map_list = list(map(lambda n: (matr.pow_matrix(n)).arr, range(1, degree + 1)))
    map_list = Matrix(map_list[0])

    return map_list


def _find_coeff(X, y, b, alpha, iter_count):
    # b0 leave unchanged
    X_with_b0 = []
    for row in X.arr:
        row[0] = 1
        X_with_b0.append(row)
    X = Matrix(X_with_b0)

    X = X.transpose()
    for epoch in range(iter_count):
        b = _make_epoch(X, y, b, alpha)

    return b


def _make_epoch(X, y, b, alpha):
    descent = multiply_matrix_and_vector(X, b)
    descent = vectors_subtraction(descent, y)
    descent = mul_vector_to_matrix(descent.transpose_vector(), X)
    descent = mul_const_with_vector(descent, 1 / X.n_rows)
    comp_alpha = mul_const_with_vector(b.sign_vector(), alpha)
    b = sum_vectors(descent, comp_alpha)

    return b

