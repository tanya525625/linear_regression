from sklearn.datasets import make_regression

from tools.matrix import Matrix
from tools.linear_regression import _find_coeff


def generate_dataset(n_samples, n_features):
    X, y = make_regression(n_samples=n_samples,
                           n_features=n_features)
    X = Matrix(X)
    y = Matrix(y)

    return X, y


if __name__  == "__main__":
    n_samples = 100
    n_features = 40

    X, y = generate_dataset(n_samples, n_features)
    # print(X.cols_count)
    # print(X.rows_count)
    # print(y.n_rows)
    print(X.n_rows)
    b = _find_coeff(X, y)