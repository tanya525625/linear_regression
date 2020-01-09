import matplotlib.pyplot as plt

import numpy as np

from sklearn import datasets, linear_model, metrics

# загрузить бостонский набор данных

boston = datasets.load_boston(return_X_y=False)

# определение матрицы признаков (X) и вектора ответа (y)

X = boston.data

y = boston.target

# разбиение X и Y на обучающие и тестовые наборы

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,

                                                    random_state=1)

# создать объект линейной регрессии

reg = linear_model.LinearRegression()

# обучаем модель с использованием тренировочных наборов
reg.fit(X_train, y_train)

# коэффициенты регрессии

print('Coefficients: \n', reg.coef_)

# дисперсионный балл: 1 означает идеальный прогноз

print('Variance score: {}'.format(reg.score(X_test, y_test)))

# график остаточной ошибки


## настройка стиля сюжета

plt.style.use('fivethirtyeight')

## построение остаточных ошибок в обучающих данных

plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train,

            color="green", s=10, label='Train data')

## построение остаточных ошибок в тестовых данных

plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test,

            color="blue", s=10, label='Test data')

## построение линии для нулевой остаточной ошибки

plt.hlines(y=0, xmin=0, xmax=50, linewidth=2)

## зарисовка легенды

plt.legend(loc='upper right')

## название сюжета

plt.title("Residual errors")

## функция показа сюжета
plt.show()
