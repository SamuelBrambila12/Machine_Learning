import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Datos de variable independiente (inches / pulgadas)
X = np.array([[1, 2], [2, 4], [3, 1], [4, 0], [5, 3]])
ax.scatter(X[:,0], X[:,1], y, c='b', marker='o')
# Datos de ejemplo (variable dependiente)
y = np.array([2, 3, 4, 5, 6])

# Parámetros de validación cruzada
num_validaciones = 10  # Número de repeticiones
test_size = 0.3  # 30% para prueba, 70% para entrenamiento
mse_scores = []

# Realizar varias divisiones y entrenar el modelo
for i in range(num_validaciones):
    # Dividir los datos aleatoriamente con 70/30
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)
    
    # Crear el modelo de regresión lineal múltiple
    modelo = LinearRegression()

    # Entrenar el modelo con los datos de entrenamiento
    modelo.fit(X_train, y_train)

    # Predecir con los datos de prueba
    y_pred = modelo.predict(X_test)

    # Calcular el error cuadrático medio (MSE) y guardarlo
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)

# Promediar el MSE de todas las validaciones
mse_promedio = np.mean(mse_scores)
print(f"Error cuadrático medio promedio (MSE): {mse_promedio}")
