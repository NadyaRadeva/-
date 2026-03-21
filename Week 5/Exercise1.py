import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def f(x):
    return a * x + b


def MSE(x_values, y_values):
    n = len(y_values)
    return sum((f(x_values[i]) - y_values[i])**2 for i in range(n)) / n


def derivative_MSE_a(x_values, y_values):
    n = len(y_values)
    return 2/n * sum((f(x_values[i]) - y_values[i]) * x_values[i] for i in range(n))


def derivative_MSE_b(x_values, y_values):
    n = len(y_values)
    return 2/n * sum((f(x_values[i]) - y_values[i]) for i in range(n))


x1 = np.linspace(-25, 25, 101)
y1 = 1/10 * sigmoid(x1) - 100


a = 0
b = 0
learning_rate = 0.0001
iterations = 10000


# training
for i in range(iterations):
    grad_a = derivative_MSE_a(x1, y1)
    grad_b = derivative_MSE_b(x1, y1)
    mse = MSE(x1, y1)

    if i % 1000 == 0:
        print(f"iter={i}, MSE={mse:.6f}, a={a:.6f}, b={b:.6f}, grad_a={grad_a:.6f}, grad_b={grad_b:.6f}")

    # update
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b


print("\nFINAL RESULT:")
print(f"a = {a}")
print(f"b = {b}")
