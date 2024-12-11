import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Regresión lineal múltiple con método del gradiente

class RegresionLinealMultiple:
    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        # Agregar una columna de unos para el término de sesgo (intercepto)
        X_con_intercepto = np.column_stack((np.ones(len(X)), X))

        # Inicializar los coeficientes aleatoriamente
        self.coef_ = np.random.randn(X_con_intercepto.shape[1])

        # Gradiente descendente
        for _ in range(self.n_iter):
            gradient = -2 * X_con_intercepto.T @ (y - X_con_intercepto @ self.coef_)
            self.coef_ -= self.learning_rate * gradient
        
        # El primer valor de coef_ corresponde al intercepto
        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]
    
    def predict(self, X):
        # Agregar una columna de unos para el término de sesgo (intercepto)
        X_con_intercepto = np.column_stack((np.ones(len(X)), X))
        return X_con_intercepto @ np.concatenate(([self.intercept_], self.coef_))
    
    def cross_validate(self, X, y, k=5):
        fold_size = len(X) // k
        mse_scores = []

        for i in range(k):
            # Crear conjuntos de entrenamiento y prueba
            start = i * fold_size
            end = (i + 1) * fold_size if i < k - 1 else len(X)

            X_train = np.concatenate((X[:start], X[end:]))
            y_train = np.concatenate((y[:start], y[end:]))
            X_test = X[start:end]
            y_test = y[start:end]

            self.fit(X_train, y_train)
            predictions = self.predict(X_test)
            mse = np.mean((y_test - predictions) ** 2)
            mse_scores.append(mse)

        return np.mean(mse_scores)

# Cargar el archivo CSV
data = pd.read_csv('baseball_hitting.csv', encoding='latin-1')

# Variables independientes (At-bat y Hits) y dependiente (AVG)
X = np.array(data[['At-bat', 'Hits']])
y = np.array(data['AVG'])

# Crear el modelo de regresión lineal múltiple
modelo = RegresionLinealMultiple(learning_rate=0.000000000019, n_iter=10000)

# Validación cruzada
mse = modelo.cross_validate(X, y)
print("MSE promedio de la validación cruzada:", mse)

# Entrenar el modelo completo
modelo.fit(X, y)

# Graficar los datos y el hiperplano resultante en 3D
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Scatter plot de los datos originales (en azul)
ax.scatter(X[:,0], X[:,1], y, c='b', marker='o', label='Datos')

# Superficie del hiperplano
xx, yy = np.meshgrid(X[:,0], X[:,1])

# Ecuación del hiperplano
zz = modelo.intercept_ + modelo.coef_[0] * xx + modelo.coef_[1] * yy
ax.plot_surface(xx, yy, zz, color='r', alpha=0.2)

# Graficar las predicciones
predicciones = modelo.predict(X)
ax.scatter(X[:,0], X[:,1], predicciones, c='g', marker='x', label='Predicciones')

# Etiquetas de los ejes
ax.set_xlabel('At-bat')
ax.set_ylabel('Hits')
ax.set_zlabel('AVG')

# Ajustar ángulos de visualización
ax.azim = 60
ax.elev = 25

# Coeficientes de la regresión (intercepto y coeficientes)
print("Intercepto: ", modelo.intercept_)
print("Coeficientes: ", modelo.coef_)

# Añadir leyenda
ax.legend()

plt.show()
