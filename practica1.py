import matplotlib.pyplot as plt
import pandas as pd

# Regresión lineal con el 100% de los datos

# Leer las columnas del CSV y almacenarlas directamente en arreglos
x = pd.read_csv('car_purchasing.csv', usecols=[5], encoding='latin-1').values.flatten()  # Variable independiente, salario anual (columna 5)
y = pd.read_csv('car_purchasing.csv', usecols=[8], encoding='latin-1').values.flatten()  # Variable dependiente, costo de carro (columna 8)

# Calcular medias
mean_x = sum(x) / len(x)
mean_y = sum(y) / len(y)

# Calcular la pendiente (m) y el intercepto (b)
numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x,y))
denominator = sum((xi - mean_x)**2 for xi in x)

m = numerator / denominator
b = mean_y - m * mean_x

# Realizar predicciones
for xi in x:
    y_pred = m * xi + b
    # Visualizar resultados
    plt.scatter(x,y,color='blue')  # Datos
    plt.scatter(xi,y_pred,color='red') # Predicciones
    plt.xlabel('Variable independiente (salario anual)')
    plt.ylabel('Variable dependiente (costo del carro)')

# Agregamos espacios vacíos, para poder agregar la leyenda, esto permite que no se repita la impresión si lo ponemos en el bucle
plt.scatter([], [], color='blue', label='Datos')  # Espacio vacío para la leyenda de Datos
plt.scatter([], [], color='red', label='Predicciones')  # Espacio vacío para la leyenda de Predicciones

# Graficar la recta de la regresión lineal
y_pred_line = m * x + b
plt.plot(x, y_pred_line, color='black', label='Regresión lineal') # Línea de regresión
plt.legend()
plt.show()

# Calcular la distancia entre cada punto y la línea de regresión
distancias = [abs(yi - m * xi - b) for xi, yi in zip(x,y)]

# Calcular el error cuadrático medio (MSE)
mse = sum((yi - m * xi - b)**2 for xi, yi in zip(x,y)) / len(x)

# Imprimir información
print('Coeficiente (pendiente):', m)
print('Intercepto:', b)
print('Distancias:', distancias)
print('Suma de las diferencias cuadradas (SSD):', sum(d**2 for d in distancias))
print('Error Cuadrático Medio (MSE):', mse)