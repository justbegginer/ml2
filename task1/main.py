import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Определение функций активации и их производных
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def leaky_relu(x, beta=0.01):
    return np.maximum(x * beta, x)


def leaky_relu_derivative(x, beta=0.01):
    return (x > x * beta).astype(float)


# Определение функций обновления весов
def gradient_descent_update(weights, bias, d_weights, d_bias, learning_rate):
    weights -= learning_rate * d_weights
    bias -= learning_rate * d_bias
    return weights, bias


def sgd_update(weights, bias, d_weights, d_bias, learning_rate):
    for i in range(len(weights)):
        weights[i] -= learning_rate * d_weights[i]
        bias -= learning_rate * d_bias
    return weights, bias


# Функция для обучения нейронной сети
def train_neural_network(X, y, activation_func, activation_derivative, optimizer_update, epochs, learning_rate):
    weights = np.zeros((X.shape[1], 1))
    bias = np.random.rand()
    losses = []

    for epoch in range(epochs):
        # Прямое распространение
        weighted_sum = np.dot(X, weights) + bias
        activated_output = activation_func(weighted_sum)

        # Обратное распространение
        error = activated_output - y
        loss = np.mean(np.square(error))
        losses.append(loss)

        # Вычисляем градиенты
        d_weighted_sum = error * activation_derivative(activated_output)
        d_weights = np.dot(X.T, d_weighted_sum)
        d_bias = np.sum(d_weighted_sum)

        # Обновляем веса и смещение
        weights, bias = optimizer_update(weights, bias, d_weights, d_bias, learning_rate)

    return weights, bias, losses


# Загружаем данные
nn_0_data = pd.read_csv('../nn_0.csv')
nn_1_data = pd.read_csv('../nn_0.csv')

# Подготовка данных
X_0 = nn_0_data.iloc[:, :-1].values
y_0 = nn_0_data.iloc[:, -1].values.reshape(-1, 1)
X_1 = nn_1_data.iloc[:, :-1].values
y_1 = nn_1_data.iloc[:, -1].values.reshape(-1, 1)

# Нормализация данных
X_0_max, X_1_max = np.max(X_0, axis=0), np.max(X_1, axis=0)
X_0, X_1 = X_0 / X_0_max, X_1 / X_1_max

# Параметры обучения
epochs = 10000
learning_rate = 0.001

activation_functions = {"relu": [relu, relu_derivative],
                        "sigmoid": [sigmoid, sigmoid_derivative],
                        "leaky_relu": [leaky_relu, leaky_relu_derivative]}
update_functions = {"sgd": sgd_update, "gradient_descent": gradient_descent_update}

for activation in activation_functions:
    for update in update_functions:
        weights_0, bias_0, losses_0 = train_neural_network(
            X_0, y_0, activation_functions[activation][0], activation_functions[activation][1],
            update_functions[update],
            epochs, learning_rate
        )
        weights_1, bias_1, losses_1 = train_neural_network(
            X_1, y_1, activation_functions[activation][0], activation_functions[activation][1],
            update_functions[update],
            epochs, learning_rate
        )

        # Вывод финальных весов, смещений и потерь
        print(f"Результаты для функции активации : {activation} и функции обновления весов: {update} для nn_0")
        print(f"Веса: {weights_0.flatten()}",
              f"Смещение: {bias_0}",
              f"Потери: {np.sum(losses_0)/len(losses_0)}",
              sep='\n')
        print(f"Результаты для функции активации : {activation} и функции обновления весов: {update} для nn_1")
        print(f"Веса: {weights_1.flatten()}",
              f"Смещение: {bias_1}",
              f"Потери: {np.sum(losses_1)/len(losses_1)}",
              sep='\n')
