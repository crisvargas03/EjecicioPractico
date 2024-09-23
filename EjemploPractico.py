# Importar las librerías necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

file_path = 'C:\\Users\\civar\\Desktop\\homeworks\\Inteligencia en Ciencia de Datos\\datos.csv'
data = pd.read_csv(file_path)

# Exploración inicial de los datos
print("Primeras filas del dataset:")
print(data.head())

# Verificar si hay valores faltantes
print("\nValores faltantes:")
print(data.isnull().sum())

# Descripción estadística del conjunto de datos
print("\nDescripción estadística:")
print(data.describe())

# Codificar la variable categórica 'Ubicacion' utilizando one-hot encoding
data_encoded = pd.get_dummies(data, columns=['Ubicacion'], drop_first=True)

# Dividir los datos en variables predictoras (X) y la variable objetivo (y)
X = data_encoded.drop('Precio', axis=1)
y = data_encoded['Precio']

# Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo utilizando el error cuadrático medio (MSE)
mse = mean_squared_error(y_test, y_pred)

# Mostrar el resultado del MSE
print(f"\nError Cuadrático Medio (MSE): {mse}")

# Mostrar las predicciones y los valores reales
print("\nPredicciones y valores reales:")
for pred, real in zip(y_pred, y_test):
    print(f"Predicción: {pred}, Real: {real}")

# Crear la gráfica Predicciones vs Valores Reales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicciones')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Línea de identidad')

# Añadir etiquetas y título
plt.title('Predicciones vs Valores Reales')
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.legend()

# Mostrar la gráfica
plt.show()
