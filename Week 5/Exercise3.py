import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


x31 = np.linspace(-50, 50, 101)
x32 = 20 * np.random.rand(101)

x3 = np.array([x31, x32]).T   # shape (101, 2)
y3 = np.zeros((101, 2))


for i in range(101):
    y3[i] = [
        sigmoid(1/10 * x3[i, 0] - 1/5 * x3[i, 1] - 1),
        sigmoid(1/3 * x3[i, 0] + 1/4 * x3[i, 1] + 3)
    ]


np.random.seed(0)
W = np.random.randn(2, 2)

learning_rate = 0.001
iterations = 20000


def f(x):
    return sigmoid(W @ x)   # matrix multiplication


def MSE(x_values, y_values):
    N = len(x_values)
    suma = 0
    for i in range(N):
        pred = f(x_values[i])
        suma += np.sum((y_values[i] - pred) ** 2)
    return suma / (2 * N)


def compute_gradient(x_values, y_values):
    N = len(x_values)
    grad = np.zeros((2, 2))

    for i in range(N):
        x = x_values[i]
        y = y_values[i]

        z = W @ x
        pred = sigmoid(z)

        error = (pred - y)
        delta = error * sigmoid_derivative(z)

        grad += np.outer(delta, x)

    return grad / N


for i in range(iterations):
    grad = compute_gradient(x3, y3)

    if i % 2000 == 0:
        print(f"iter={i}, MSE={MSE(x3, y3):.6f}")
        print("W =", W)

    W -= learning_rate * grad


print("\nFINAL RESULT:")
print(W)
