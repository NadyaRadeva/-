import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# model (FIXED to match y2)
def f(x):
    return 1/10 * sigmoid(c * sigmoid(a * x + b) + d) - 100


def MSE(x_values, y_values):
    n = len(y_values)
    return sum((y_values[i] - f(x_values[i]))**2 for i in range(n)) / n


# derivatives
def derivative_MSE_a(x_values, y_values):
    n = len(y_values)
    suma = 0
    for i in range(n):
        inner = sigmoid(a * x_values[i] + b)
        outer = sigmoid(c * inner + d)

        suma += (y_values[i] - (1/10 * outer - 100)) * (1/10) * c * inner * (1 - inner) * x_values[i] * outer * (1 - outer)

    return -2 / n * suma


def derivative_MSE_b(x_values, y_values):
    n = len(y_values)
    suma = 0
    for i in range(n):
        inner = sigmoid(a * x_values[i] + b)
        outer = sigmoid(c * inner + d)

        suma += (y_values[i] - (1/10 * outer - 100)) * (1/10) * c * inner * (1 - inner) * outer * (1 - outer)

    return -2 / n * suma


def derivative_MSE_c(x_values, y_values):
    n = len(y_values)
    suma = 0
    for i in range(n):
        inner = sigmoid(a * x_values[i] + b)
        outer = sigmoid(c * inner + d)

        suma += (y_values[i] - (1/10 * outer - 100)) * (1/10) * inner * outer * (1 - outer)

    return -2 / n * suma


def derivative_MSE_d(x_values, y_values):
    n = len(y_values)
    suma = 0
    for i in range(n):
        outer = sigmoid(c * sigmoid(a * x_values[i] + b) + d)

        suma += (y_values[i] - (1/10 * outer - 100)) * (1/10) * outer * (1 - outer)

    return -2 / n * suma


x2 = np.linspace(-25, 25, 101)
y2 = 1/10 * sigmoid(3 * x2 - 2) - 100


np.random.seed(0)
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

learning_rate = 0.1
iterations = 50000


for i in range(iterations):
    grad_a = derivative_MSE_a(x2, y2)
    grad_b = derivative_MSE_b(x2, y2)
    grad_c = derivative_MSE_c(x2, y2)
    grad_d = derivative_MSE_d(x2, y2)

    if i % 5000 == 0:
        mse = MSE(x2, y2)
        print(f"iter={i}, MSE={mse:.6f}, a={a:.4f}, b={b:.4f}, c={c:.4f}, d={d:.4f}")

    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d


print("\nFINAL RESULT:")
print(f"a = {a}")
print(f"b = {b}")
print(f"c = {c}")
print(f"d = {d}")
