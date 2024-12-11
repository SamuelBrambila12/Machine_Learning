import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

class KMeans:
    def __init__(self, n_clusters, max_iter=1300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        # Paso 1: Inicialización de los centroides
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        
        for _ in range(self.max_iter):
            # Paso 2: Asignar etiquetas al centroide más cercano
            labels = self._assign_labels(X)
            
            # Paso 3: Actualizar los centroides
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
            
            # Comprobar convergencia
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids
        
        self.labels_ = labels
        return self

    def _assign_labels(self, X):
        # Calcular distancias de cada punto a cada centroide
        distances = np.sqrt(((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)

# Funciones de distancia
def distancia_manhattan(p, q):
    return np.sum(np.abs(q - p))

def distancia_minkowski(p, q, p_value):
    return np.power(np.sum(np.power(np.abs(q - p), p_value)), 1 / p_value)

def distancia_chebyshev(p, q):
    return np.max(np.abs(q - p))

# Cargar el conjunto de datos de Iris
data = load_iris()
X = data.data 

# Aplicar K-Means con 3, 5 y 10 centroides
for n_clusters in [3, 5, 10]:
    kmeans_custom = KMeans(n_clusters=n_clusters)
    kmeans_custom.fit(X)
    centroids = kmeans_custom.centroids
    labels = kmeans_custom.labels_

    # Calcular e imprimir distancias entre los centroides
    print(f"\nDistancias para {n_clusters} centroides:")
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            dist_manhattan = distancia_manhattan(centroids[i], centroids[j])
            dist_minkowski = distancia_minkowski(centroids[i], centroids[j], p_value=2)  # Euclidiana
            dist_chebyshev = distancia_chebyshev(centroids[i], centroids[j])
            print(f"Centroides {i+1} y {j+1}:")
            print(f" - Distancia Manhattan: {dist_manhattan}")
            print(f" - Distancia Euclidiana (Minkowski con p=2): {dist_minkowski}")
            print(f" - Distancia Chebyshev: {dist_chebyshev}")

    # Visualización con cuadro de simbología
    plt.figure()
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.5, label='Centroides')
    plt.title(f'Clustering K-Means con {n_clusters} Centroides')
    plt.xlabel(data.feature_names[0])
    plt.ylabel(data.feature_names[1])

    # Añadir leyenda para poder visualizar de manera más sencilla los clústeres
    if n_clusters <= 3:
        species_names = ["Setosa", "Versicolor", "Virginica"]
        legend_labels = species_names[:n_clusters]
    else:
        legend_labels = [f'Clúster {i+1}' for i in range(n_clusters)]
    handles, _ = scatter.legend_elements()
    plt.legend(handles=handles, labels=legend_labels + ['Centroides'], title="Clusters")
    plt.show()