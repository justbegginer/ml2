import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD

# Загрузка и предобработка данных MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Конвертация меток в one-hot encoding
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Создание модели нейронной сети
model = Sequential()
model.add(Flatten(input_shape=(28, 28, 1))) # Преобразование входных данных в вектор
model.add(Dense(128, activation='relu'))    # Первый скрытый слой
model.add(Dense(10, activation='softmax'))  # Выходной слой с softmax для классификации

# Компиляция модели
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Обучение модели
history = model.fit(train_images, train_labels,
                    epochs=5,
                    batch_size=32,
                    validation_data=(test_images, test_labels))

# Оценка модели
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Тестовая точность: {test_acc}\n'
      f'Тестовые потери: {test_loss}')

# Визуализация истории обучения
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig("accuracy.png")
