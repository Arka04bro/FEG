import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def dms_to_decimal(dms_str):
    parts = dms_str.split()
    degrees = float(parts[0][:-1])
    minutes = float(parts[1][:-1]) if len(parts) > 1 else 0
    seconds = float(parts[2][:-1]) if len(parts) > 2 else 0
    direction = parts[-1]
    decimal = degrees + minutes / 60 + seconds / 3600
    if direction in ['с.ш.', 'ю.ш.']:
        decimal = -decimal if direction == 'ю.ш.' else decimal
    return decimal

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    features = [
        'Широта', 'Долгота', 'Скорость ветра', 
        'Направление ветра', 'Температура', 
        'Влажность', 'Топография местности', 
        'Технические и искусственные барьеры'
    ]
    data['Широта'] = data['Широта'].apply(dms_to_decimal)
    data['Долгота'] = data['Долгота'].apply(dms_to_decimal)
    X = data[features]
    y_speed = data['Скорость распространения пожара']
    y_direction = data['Направление фронта пожара']
    X = pd.get_dummies(X, columns=['Направление ветра', 'Технические и искусственные барьеры'], drop_first=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y_speed, y_direction

def build_model(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

file_path = "C:/Users/zhien/Downloads/Новая таблица - Лист1.csv"
data = load_data(file_path)
X, y_speed, y_direction = preprocess_data(data)
y = np.column_stack((y_speed, y_direction))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = build_model(X_train.shape[1])
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
loss, mae = model.evaluate(X_test, y_test)
print(f'Потери на тестовой выборке: {loss}, Средняя абсолютная ошибка: {mae}')
predictions = model.predict(X_test)
print("Предсказания на тестовой выборке:")
print(predictions)

def predict_user_input():
    latitude = dms_to_decimal(input("Введите широту (например, '52°27'31'' с.ш.'): "))
    longitude = dms_to_decimal(input("Введите долготу (например, '64°00'40'' в.д.'): "))
    wind_speed = float(input("Введите скорость ветра (км/ч): "))
    wind_direction = input("Введите направление ветра (например, 'северное'): ")
    temperature = float(input("Введите температуру (°C): "))
    humidity = float(input("Введите влажность (%): "))
    topography = float(input("Введите топографию местности (%): "))
    barriers = input("Введите технические и искусственные барьеры (например, 'Дороги, просеки'): ")
    user_data = pd.DataFrame([[latitude, longitude, wind_speed, wind_direction, temperature, humidity, topography, barriers]], columns=[
        'Широта', 'Долгота', 'Скорость ветра', 
        'Направление ветра', 'Температура', 
        'Влажность', 'Топография местности', 
        'Технические и искусственные барьеры'
    ])
    user_data = pd.get_dummies(user_data, columns=['Направление ветра', 'Технические и искусственные барьеры'], drop_first=True)
    scaler = StandardScaler()
    user_data = scaler.fit_transform(user_data)
    prediction = model.predict(user_data)
    print("Прогнозируемая скорость распространения пожара и направление фронта пожара:")
    print(prediction)

predict_user_input()
