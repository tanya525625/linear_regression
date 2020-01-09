from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

from tools.matrix import Matrix
from tools.linear_regression import _find_coeff
from tools.linear_regression import LinearRegression


def generate_dataset(n_samples, n_features):
    X, y = make_regression(n_samples=n_samples,
                           n_features=n_features)

    if n_features == 1:
        X = [float(el) for el in X]
    X = Matrix(X)
    y = Matrix(y)

    return X, y


if __name__  == "__main__":
    n_samples = 10
    n_features = 1

    X, y = generate_dataset(n_samples, n_features)
    # print(X.cols_count)
    # print(X.rows_count)
    # print(y.n_rows)
    # print(X.n_rows)

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # построение линии регрессии
    # plt.plot(X.arr, y_pred, color="g")

