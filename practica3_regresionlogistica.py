import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, KFold

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.losses = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            loss = self.loss(y_predicted, y)
            self.losses.append(loss)
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls

    def cross_validation(self, X, y, k=5):
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        accuracies, precisions, recalls = [], [], []
        
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            self.fit(X_train, y_train)

            y_pred = self.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)

            accuracies.append(acc)
            precisions.append(prec)
            recalls.append(rec)
        
        print(f"Resultados con validación cruzada de {k} folds:")
        print(f"Accuracy (Exactitud): {np.mean(accuracies):.4f}")
        print(f"Precision (Precisión): {np.mean(precisions):.4f}")
        print(f"Recall (Exhaustividad): {np.mean(recalls):.4f}")

    def plot_sigmoid_with_predictions(self, X, y):
        # Graficar la curva sigmoide
        z = np.linspace(-10, 10, 100)
        sigmoid_curve = self.sigmoid(z)
        plt.plot(z, sigmoid_curve, label="Curva Sigmoide", color='b')

        # Predecir valores usando el conjunto X
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)

        # Convertir las predicciones en clases (0 o 1)
        y_predicted_cls = np.array([1 if i > 0.5 else 0 for i in y_predicted])

        # Graficar los puntos predichos (0 o 1) con los colores correspondientes a las etiquetas
        plt.scatter(linear_model, y_predicted_cls, c=y, edgecolors='k', marker='o', label='Datos Predichos', cmap=plt.cm.RdBu)

        # Título y etiquetas del gráfico
        plt.title("Regresión logística")
        plt.xlabel("Combinación Lineal (X * weights + bias)")
        plt.ylabel("Predicción (0: Benigno, 1: Maligno)")
        plt.yticks([0, 1])  # Mostrar solo 0 y 1 en el eje y
        plt.grid(True)
        plt.legend()
        plt.show()

# Cargar el dataset Breast Cancer de scikit-learn
cancer = datasets.load_breast_cancer()
X = cancer.data[:, [0, 8]]  # "worst radius" (col 0) y "mean concavity" (col 8)
y = (cancer.target != 0) * 1 # Clases (0: benigno, 1: maligno)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear y entrenar el modelo
model = LogisticRegression(learning_rate=0.01, num_iterations=10000)
model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular la matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print("Matriz de confusión")
print(cm)

# Visualizar la matriz de confusión
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)
plt.title('Matriz de Confusión')
plt.colorbar()
plt.xticks([0, 1], ['Benigno', 'Maligno'])
plt.yticks([0, 1], ['Benigno', 'Maligno'])
plt.xlabel('Etiqueta Predicha')
plt.ylabel('Etiqueta Verdadera')
plt.show()

# Realizar validación cruzada con K-folds
model.cross_validation(X, y, k=5)

# Visualizar la curva sigmoide y los valores predichos
model.plot_sigmoid_with_predictions(X_test, y_test)