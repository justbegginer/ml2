import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential

# Загрузка данных
nn_0_df = pd.read_csv('../nn_0.csv')
nn_1_df = pd.read_csv('../nn_1.csv')

# Нормализация меток классов для использования с сигмоидальной функцией активации
y_0 = (nn_0_df['class'].values + 1) / 2
y_1 = (nn_1_df['class'].values + 1) / 2


# Функция для создания и обучения модифицированной модели
def create_and_train_model(X, y, activation_func_name, optimization_func, epochs=1000):
    model = Sequential([
        Input(shape=(2,)),
        Dense(1, activation=activation_func_name)  # Выходной слой
    ])

    model.compile(optimizer=optimization_func(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Ранняя остановка для предотвращения переобучения
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Разделение данных на обучающую и валидационную выборки
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, verbose=0,
                        callbacks=[early_stopping])
    return model, history


# Применение функции к датасету nn_1
for activation_func in ['sigmoid', 'relu', 'leaky_relu', 'tanh']:
    for optimization_func in [SGD, Adam, RMSprop]:
        model0, history0 = create_and_train_model(nn_0_df[['X1', 'X2']].values, y_0, activation_func, optimization_func)
        model1, history1 = create_and_train_model(nn_1_df[['X1', 'X2']].values, y_1, activation_func, optimization_func)

    # Вывод информации о производительности модели
        print(f"{activation_func}, {optimization_func}")
        print(f"Ошибка на валидационном датасете: nn_0 - {min(history0.history['val_loss'])}; nn_1  -  {min(history1.history['val_loss'])}")
        print(f"Точность на валидационном датасете: nn_0-{max(history1.history['val_accuracy'])}; nn_1-{min((history1.history['val_accuracy']))}")
