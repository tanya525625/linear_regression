from sklearn.datasets import make_regression
import pylab
import matplotlib.pyplot as plt

from math import log

from tools.matrix import Matrix
from tools.linear_regression import LinearRegression
from tools.ridge_regularisation import RidgeRegularisation
from tools.lasso_regularisation import LassoRegularisation
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
    y_test = Matrix(y.arr[end_train_point:])

    return X_train, y_train, X_test, y_test


def plot_regression_line(x: Matrix, y: Matrix, y_pred: list):
    if x.n_cols == 1:
        # starting position of points
        print(len(y_pred))
        plt.scatter(x.arr, y.arr, color="c")
        # regression line
        plt.plot(x.arr, y_pred, color="g")

        plt.show()


def analyze_ridge_regularisation(min_alpha, max_alpha):
    alpha_list = list(range(min_alpha, max_alpha))
    # alpha_list = [log(a) for a in alpha_list]
    metrics_list = []
    weight_list = []

    for alpha in alpha_list:
        model = RidgeRegularisation(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        metrics_list.append(rmse(y_pred=y_pred, y_true=y_test.arr))
        weight_list.append(model.b.find_mean_in_vect())

    pylab.figure(1)
    plt.plot(alpha_list, metrics_list, color="r")
    plt.title("Ridge Regularisation metrics")
    plt.xlabel("log(alpha)")
    plt.ylabel("RMSE")

    pylab.figure(2)
    plt.title("Ridge Regularisation Weight coeff")
    plt.xlabel("log(alpha)")
    plt.ylabel("Weight coeff")
    plt.plot(alpha_list, weight_list, color="c")

    plt.show()


def analyze_lasso_regularisation(min_alpha, max_alpha, learning_rate, degree, iter_count):
    alpha_list = list(range(min_alpha, max_alpha))
    metrics_list = []
    weight_list = []
    alpha_list = [log(a) for a in alpha_list]

    for alpha in alpha_list:
        model = LassoRegularisation(alpha=alpha, learning_rate=learning_rate,
                                    degree=degree, iter_count=iter_count)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        metrics_list.append(rmse(y_pred=y_pred, y_true=y_test.arr))
        weight_list.append(model.b.find_mean_in_vect())

    pylab.figure(1)
    plt.plot(alpha_list, metrics_list, color="r")
    plt.title("Lasso Regularisation metrics")
    plt.xlabel("log(alpha)")
    plt.ylabel("RMSE")

    pylab.figure(2)
    plt.title("Lasso Regularisation Weight coeff")
    plt.xlabel("log(alpha)")
    plt.ylabel("Weight coeff")
    plt.plot(alpha_list, weight_list, color="c")

    plt.show()


if __name__  == "__main__":
    n_samples = 350
    n_features = 1

    X, y = generate_dataset(n_samples, n_features)
    X_train, y_train, X_test, y_test = split_test_and_train(X, y, 0.3)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    plot_regression_line(X_train, y_train, y_pred)
    metric = rmse(y_pred=y_pred, y_true=y_test.arr)
    print(f'RMSE for linear regression = {metric}')

    min_alpha = 1
    max_alpha = 50
    analyze_ridge_regularisation(min_alpha, max_alpha)

    min_alpha = 1
    max_alpha = 50
    learning_rate = 0.1
    degree = 4
    iter_count = 100
    analyze_lasso_regularisation(min_alpha, max_alpha, learning_rate, degree, iter_count)











