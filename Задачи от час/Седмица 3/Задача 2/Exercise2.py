import tensorflow as tf
import numpy as np

X = np.array([
    [3, 5, 2],
    [-2, 4, 6],
    [-3, -1, 5],
    [4, -3, 1],
    [2, 1, -5],
    [-4, 3, -2],
    [-1, -2, -3],
    [3, -4, -6]
])

def octant(p):
    x, y, z = p

    if x>0 and y>0 and z>0: return 0
    if x<0 and y>0 and z>0: return 1
    if x<0 and y<0 and z>0: return 2
    if x>0 and y<0 and z>0: return 3
    if x>0 and y>0 and z<0: return 4
    if x<0 and y>0 and z<0: return 5
    if x<0 and y<0 and z<0: return 6
    if x>0 and y<0 and z<0: return 7

y = np.array([octant(p) for p in X])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(12, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(8, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Training brain... 🧠")
model.fit(X, y, epochs=300, verbose=0)

test = np.array([
    [4, 2, 1],
    [-5, 3, 2],
    [-1, -4, 7],
    [2, -3, 9],
    [1, 5, -2],
    [-3, 1, -4],
    [-2, -3, -5],
    [5, -6, -3]
])

pred = model.predict(test)
classes = np.argmax(pred, axis=1)

print("\nPredictions:\n")

for p, c, probs in zip(test, classes, pred):

    print(f"Point {p} → octant {c+1}")

    print(" probabilities:")
    for i in range(8):
        print(f"   O{i+1}: {probs[i]*100:.2f}%")

    print()
