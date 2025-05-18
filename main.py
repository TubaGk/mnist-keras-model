from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Input

(x_train, y_train), (x_test, y_test) = mnist.load_data()# 60k eğitim 10k test

x_train, x_test = x_train / 255.0, x_test / 255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

import matplotlib.pyplot as plt
import numpy as np

num_samples = 10
x_test_samples = x_test[:num_samples]
y_test_samples = y_test[:num_samples]

predictions = model.predict(x_test_samples)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test_samples, axis=1)

# 1 satır, num_samples sütunluk bir görsel ızgarası
fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2))

for i in range(num_samples):
    ax = axes[i]
    ax.imshow(x_test_samples[i], cmap='gray')
    ax.set_title(f"Tahmin:{predicted_classes[i]}\nGerçek:{true_classes[i]}")
    ax.axis('off')

plt.tight_layout()
plt.show()
