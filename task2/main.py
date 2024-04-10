import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential

# Загрузка данных
nn_0_df = pd.read_csv('../nn_0.csv')
nn_1_df = pd.read_csv('../nn_1.csv')

# Нормализация меток классов для использования с сигмоидальной функцией активации
y_0 = (nn_0_df['class'].values + 1) / 2
y_1 = (nn_1_df['class'].values + 1) / 2


# Функция для создания и обучения модифицированной модели
def create_and_train_model(X, y, epochs=10000):
    model = Sequential([
        Input(shape=(2,)),
        Dense(1, activation='leaky_relu'),  # Первый скрытый слой
        Dropout(0.5),  # Добавление ещё одного слоя Dropout
        Dense(1, activation='sigmoid')  # Выходной слой
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    # Ранняя остановка для предотвращения переобучения
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Разделение данных на обучающую и валидационную выборки
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, verbose=0,
                        callbacks=[early_stopping])
    return model, history


# Применение функции к датасету nn_1
model, history = create_and_train_model(nn_1_df[['X1', 'X2']].values, y_1)

# Вывод информации о производительности модели
print(f"Минимальная ошибка на валидационном датасете: {min(history.history['val_loss'])}")
print(f"Лучшая точность на валидационном датасете: {max(history.history['val_accuracy'])}")
