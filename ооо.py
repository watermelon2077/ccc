import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')

# Предварительный анализ данных
print("Первые 5 строк данных:")
print(data.head())

print("\nИнформация о данных:")
print(data.info())

data.dropna(inplace=True)

# Преобразование типов данных
data['Date'] = pd.to_datetime(data['Date'])

# Подсчет статистики
print("\nСтатистика по числовым данным:")
print(data.describe())

# Визуализация данных
plt.figure(figsize=(10, 6))

# График временного ряда
plt.subplot(2, 1, 1)
plt.plot(data['Date'], data['Value'], color='blue')
plt.title('График временного ряда')
plt.xlabel('Дата')
plt.ylabel('Значение')

# Гистограмма значений
plt.subplot(2, 1, 2)
plt.hist(data['Value'], bins=20, color='green', alpha=0.7)
plt.title('Гистограмма значений')
plt.xlabel('Значение')
plt.ylabel('Частота')

plt.tight_layout()
plt.show()

# Вычисление корреляции между переменными
correlation = data.corr()
print("\nКорреляция между переменными:")
print(correlation)

# Построение тепловой карты корреляции
plt.figure(figsize=(8, 6))
plt.title('Тепловая карта корреляции')
plt.imshow(correlation, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=45)
plt.yticks(range(len(correlation.columns)), correlation.columns)
plt.show()

# Применение машинного обучения для прогнозирования
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Определение признаков и целевой переменной
X = data[['Feature1', 'Feature2']]
y = data['Target']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Оценка качества модели
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print("\nОценка качества модели:")
print(f"Коэффициент детерминации на обучающей выборке: {train_score}")
print(f"Коэффициент детерминации на тестовой выборке: {test_score}")

# Прогнозирование значений
y_pred = model.predict(X_test)

# Оценка качества прогнозирования
mse = mean_squared_error(y_test, y_pred)
print(f"\nСреднеквадратичная ошибка: {mse}")
