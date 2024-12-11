import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Regresión lineal múltiple usando librerías de Python

# Cargar el archivo CSV
data = pd.read_csv('baseball_hitting.csv', encoding='latin-1')

# Se forma el arreglo a partir de los datos At-bat (turnos de bateo conseguidos) y Hits (Hits conseguidos), Variables independientes
X = np.array(data[['At-bat', 'Hits']])

# Para la variable dependiente, se usará AVG (promedio de bateo o average)
y = np.array(data['AVG'])

# Obtener correlación entre las variables independientes y la dependiente
correlaciones = data[['At-bat', 'Hits', 'AVG']].corr()
print("Correlación entre variables independientes y dependiente:")
print(correlaciones)

# Definir el número de repeticiones para la validación cruzada
repeticiones = 5
mse_resultados = []
y_pred_total = []  # Almacenar todas las predicciones

for i in range(repeticiones):
    # Dividir los datos (70% para entrenamiento, 30% para prueba)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
    
    # Crear el modelo de regresión lineal múltiple
    modelo = LinearRegression()
    
    # Entrenar el modelo con los datos de entrenamiento
    modelo.fit(X_train, y_train)
    
    # Hacer predicciones en el conjunto de prueba
    y_pred = modelo.predict(X_test)
    y_pred_total.append(y_pred)  # Guardar las predicciones
    
    # Calcular el error cuadrático medio (MSE)
    mse = mean_squared_error(y_test, y_pred)
    mse_resultados.append(mse)
    print(f"MSE en iteración {i+1}: {mse}")

# Mostrar el resultado promedio del MSE en las repeticiones
print(f"Promedio de MSE: {np.mean(mse_resultados)}")
print(f"Desviación estándar de MSE: {np.std(mse_resultados)}")

# Entrenar el modelo con todos los datos para imprimir los coeficientes finales
modelo.fit(X, y)

# Imprimir el intercepto y los coeficientes
print(f"Intercepto: {modelo.intercept_}")
print(f"Coeficientes: {modelo.coef_}")

# Graficar los datos y el hiperplano resultante en 3D (usando los datos completos)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Scatter plot de los datos originales (en azul)
ax.scatter(X[:,0], X[:,1], y, c='b', marker='o', label='Datos')

# Superficie del hiperplano
xx, yy = np.meshgrid(X[:,0], X[:,1])

# Ecuación del hiperplano
zz = modelo.intercept_ + modelo.coef_[0] * xx + modelo.coef_[1] * yy
ax.plot_surface(xx, yy, zz, color='r', alpha=0.2)

# Graficar las predicciones en otro color (en verde)
X_test_flat = np.concatenate([X_test for _ in range(repeticiones)])  # Concatenar los conjuntos de prueba
y_pred_flat = np.concatenate(y_pred_total)  # Concatenar todas las predicciones
ax.scatter(X_test_flat[:, 0], X_test_flat[:, 1], y_pred_flat, c='g', marker='x', label='Predicciones')

# Etiquetas de los ejes
ax.set_xlabel('At-bat')
ax.set_ylabel('Hits')
ax.set_zlabel('AVG')

# Ajustar ángulos de visualización
ax.azim = 30
ax.elev = 20

# Añadir leyenda
ax.legend()

plt.show()
