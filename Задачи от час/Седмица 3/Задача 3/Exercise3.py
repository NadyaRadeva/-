import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = np.arange(0, 48.01, 0.25)
y = np.sin(X)

X = X.reshape(-1,1)
y = y.reshape(-1,1)

X_test = np.arange(0, 48.01, 0.15).reshape(-1,1)
y_true = np.sin(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='tanh', input_shape=(1,)),
    tf.keras.layers.Dense(32, activation='tanh'),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse'
)

print("Training brain... 🧠")
model.fit(X, y, epochs=400, verbose=0)

y_pred = model.predict(X_test)

print("\nSample predictions:\n")
for i in range(0, len(X_test), 20):
    print(f"x={X_test[i][0]:.2f}  sin(x)≈{y_pred[i][0]:.4f}")

plt.figure(figsize=(10,5))

plt.plot(X_test, y_true, label="true sin(x)")
plt.plot(X_test, y_pred, '--', label="NN approximation")

plt.legend()
plt.grid()
plt.title("Neural Network approximation of sin(x)")
plt.show()
