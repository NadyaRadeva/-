import tensorflow as tf
import numpy as np

X = np.array([
    [1, 2], [1, 5], [-2, 3], [-3, 2],
    [4, -5], [5, -12], [-6, -3], [-7, 10],
    [8, -4], [2, -9]
])

def quadrant(p):
    x, y = p
    if x > 0 and y > 0:
        return 0
    if x < 0 and y > 0:
        return 1
    if x < 0 and y < 0:
        return 2
    if x > 0 and y < 0:
        return 3

y = np.array([quadrant(p) for p in X])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(2,)), 
    tf.keras.layers.Dense(4, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Training the brain... 🧠")
model.fit(X, y, epochs=200, verbose=0)

test_points = np.array([
    [3, 2], [-1, -3], [2, 3], [3, 12],
    [-4, 5], [2, -1], [4, -3], [7, 1],
    [3, -4], [2, 2]
])

pred = model.predict(test_points)

print("\n🔮 Predictions:\n")

for point, probs in zip(test_points, pred):

    quadrant = np.argmax(probs) + 1

    print(f"Point {point} → quadrant {quadrant}")

    print(" probabilities:")
    print(f"   Q1: {probs[0]*100:.2f}%")
    print(f"   Q2: {probs[1]*100:.2f}%")
    print(f"   Q3: {probs[2]*100:.2f}%")
    print(f"   Q4: {probs[3]*100:.2f}%")
    print()
