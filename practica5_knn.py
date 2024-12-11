import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from collections import Counter

# Cargar el dataset
penguins = sns.load_dataset("penguins")

# Eliminar filas con valores faltantes
penguins.dropna(inplace=True)

# Datos cuantitativos
# Variables independientes (input)
penguins.describe()

# Datos cualitativos
# Variable dependiente (output), en este caso se elige a "Species"
penguins['species'].unique()
penguins['species'].value_counts().plot(kind='bar')
plt.show()

# Convertir la variable 'sex' a valores numéricos
penguins['sex'] = penguins['sex'].map({'Male': 0, 'Female': 1})

# Convertir la variable 'island' usando codificación de etiquetas
penguins['island'] = penguins['island'].astype('category').cat.codes

# Mapa de correlación
penguins_numeric = penguins.select_dtypes(include=[np.number])
sns.heatmap(penguins_numeric.corr(), square=True, annot=True)
plt.title('Mapa de correlación')
plt.show()

# Graficar las 4 variables independientes más importantes
fig = px.scatter_3d(penguins, 
                    x="body_mass_g", 
                    y="flipper_length_mm", 
                    z="bill_length_mm", 
                    size="bill_depth_mm",       # Profundidad del pico (cuarta variable representada por el tamaño de los puntos)
                    color="species",            # Colorear según la especie del pingüino
                    color_discrete_map = {"Joly": "blue", "Bergeron": "violet", "Coderre":"pink"}
                    )
fig.show()

class KNN:
    def __init__(self, k=3):
        self.k = k

    # Función para calcular la distancia euclidiana
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

    # Método para entrenar el modelo
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    # Método para realizar predicciones
    def predict(self, X_test):
        predictions = []
        for test_point in X_test:
            # Calcular la distancia entre el punto de prueba y todos los puntos de entrenamiento
            distances = [self.euclidean_distance(test_point, x_train) for x_train in self.X_train]
            # Ordenar los índices de las distancias más cercanas
            k_indices = np.argsort(distances)[:self.k]
            # Obtener las etiquetas de los vecinos más cercanos
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            # Asignar la clase más común entre los vecinos más cercanos
            most_common = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(most_common)
        return predictions

    # Función para dividir los datos de entrenamiento y prueba
    def train_test_split(self, X, y, test_size=0.3, random_state=42):
        if random_state is not None:
            np.random.seed(random_state)
        # Mezclar los datos aleatoriamente
        indices = np.random.permutation(len(X))
        test_size = int(len(X) * test_size)
        # Dividir los datos
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

    # Función para encontrar el mejor número de vecinos (k)
    def find_best_k(self, X_train, y_train, X_test, y_test, max_k=15):
        train_accuracy = []
        test_accuracy = []
        for k in range(1, max_k + 1):
            self.k = k
            # Entrenar el modelo con los datos de entrenamiento
            self.fit(X_train, y_train)

            # Predecir para los datos de entrenamiento y prueba
            y_train_pred = self.predict(X_train)  # Solo X_train
            y_test_pred = self.predict(X_test)    # Solo X_test

            # Calcular la precisión para los datos de entrenamiento y prueba
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)

            train_accuracy.append(train_acc)
            test_accuracy.append(test_acc)

        # Graficar la precisión en función de k
        plt.plot(range(1, max_k + 1), train_accuracy, label='Training Accuracy')
        plt.plot(range(1, max_k + 1), test_accuracy, label='Test Accuracy')
        plt.xlabel('n_neighbors')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. n_neighbors')
        plt.legend()
        plt.show()

    # Función para realizar validación cruzada
    def cross_validate(self, X, y, folds=5):
        fold_size = len(X) // folds
        scores = []
        for i in range(folds):
            # Crear los conjuntos de entrenamiento y prueba para cada pliegue
            X_train = np.concatenate([X[:i * fold_size], X[(i + 1) * fold_size:]], axis=0)
            y_train = np.concatenate([y[:i * fold_size], y[(i + 1) * fold_size:]], axis=0)
            X_test = X[i * fold_size:(i + 1) * fold_size]
            y_test = y[i * fold_size:(i + 1) * fold_size]

            # Entrenar el modelo con los datos de entrenamiento
            self.fit(X_train, y_train)

            # Realizar predicciones sobre los datos de prueba
            y_pred = self.predict(X_test)

            # Calcular la precisión y añadirla a la lista de puntuaciones
            score = accuracy_score(y_test, y_pred)
            scores.append(score)

        # Mostrar las precisiones por pliegue y la media
        print(f'Precisión por cada iteración: {scores}')
        print(f'Precisión promedio: {np.mean(scores)}')

    # Función para graficar la matriz de confusión
    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False,
                    xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
        plt.xlabel('Predicho')
        plt.ylabel('Verdadero')
        plt.title('Matriz de Confusión')
        plt.show()

# Cargar los datos
X = penguins[['body_mass_g', 'flipper_length_mm', 'bill_length_mm', 'bill_depth_mm']].values
y = penguins['species'].values

# Dividir los datos en entrenamiento y prueba
knn = KNN(k=13)
X_train, X_test, y_train, y_test = knn.train_test_split(X, y)

# Entrenar el modelo
knn.fit(X_train, y_train)

# Realizar predicciones
y_pred = knn.predict(X_test)

# Realizar validación cruzada
knn.cross_validate(X, y, folds=5)

# Calcular las métricas
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))
# Graficar la matriz de confusión
knn.plot_confusion_matrix(y_test, y_pred)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Encontrar el mejor número de vecinos (k)
knn.find_best_k(X_train, y_train, X_test, y_test)