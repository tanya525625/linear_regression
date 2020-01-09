from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

from tools.matrix import Matrix
from tools.linear_regression import LinearRegression
from tools.l2_regularisation import L2Regularisation
from tools.functions import rmse


def generate_dataset(n_samples, n_features):
    X, y = make_regression(n_samples=n_samples,
                           n_features=n_features, noise=20)

    if n_features == 1:
        X = [float(el) for el in X]
    X = Matrix(X)
    y = Matrix(y)

    return X, y


def split_test_and_train(X: Matrix, y: Matrix, proportion):
    end_train_point = int(X.n_rows * proportion)
    X_train = Matrix(X.arr[:end_train_point])
    y_train = Matrix(y.arr[:end_train_point])
    X_test = Matrix(X.arr[end_train_point:])
    y_test = Matrix(X.arr[end_train_point:])

    return X_train, y_train, X_test, y_test


def plot_regression_line(x, y, y_pred):
    # starting position of points
    plt.scatter(x, y, color="c")
    # regression line
    plt.plot(x, y_pred, color="g")
    plt.show()


def analyze_regularisation(min_alpha, max_alpha):
    alpha_list = list(range(min_alpha, max_alpha))
    metrics_list = []

    for alpha in alpha_list:
        l2r = L2Regularisation(alpha)
        l2r.fit(X_train, y_train)
        y_pred = l2r.predict(X_train)
        metrics_list.append(rmse(y_pred=y_pred, y_true=y_test.arr))

    plt.plot(alpha_list, metrics_list, color="r")
    plt.show()


if __name__  == "__main__":
    n_samples = 350
    n_features = 1

    X, y = generate_dataset(n_samples, n_features)
    X_train, y_train, X_test, y_test = split_test_and_train(X, y, 0.3)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    plot_regression_line(X_train.arr, y_train.arr, y_pred)
    metric = rmse(y_pred=y_pred, y_true=y_test.arr)
    print(f'RMSE for linear regression = {metric}')

    min_alpha = -90
    max_alpha = 100
    analyze_regularisation(min_alpha, max_alpha)


