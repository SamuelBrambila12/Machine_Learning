# Tratamiento de datos
# ==============================================================================
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

# Gráficos y métricas
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_predict

# Configuración matplotlib
# ==============================================================================
plt.rcParams['image.cmap'] = "bwr"
plt.rcParams['savefig.bbox'] = "tight"
style.use('ggplot')

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

# Carga del Dataset Breast Cancer Wisconsin
# ==============================================================================
data = load_breast_cancer()
X = pd.DataFrame(data['data'], columns=data['feature_names'])
y = data['target']

# Usaremos dos características para graficar
X = X[['mean radius', 'mean texture']]

# Visualización de los datos
# ==============================================================================
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(X['mean radius'], X['mean texture'], c=y)
ax.set_title("Breast Cancer Dataset (Radio Medio vs Textura Media)")
plt.show()

# División de los datos en train y test
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=0.7,
    random_state=1234,
    shuffle=True
)

# Parámetros para la validación cruzada
# ==============================================================================
param_grid = {'C': [0.1, 1, 10, 100, 1000]}
modelo_svm = SVC(kernel='linear', random_state=123)

# Implementación de validación cruzada con StratifiedKFold
# ==============================================================================
cv = StratifiedKFold(n_splits=5)

# GridSearchCV para buscar el mejor modelo
# ==============================================================================
grid_search = GridSearchCV(estimator=modelo_svm, param_grid=param_grid, scoring='accuracy', cv=cv)
grid_search.fit(X_train, y_train)

# Evaluación para cada valor de C en la validación cruzada con K-folds
# ==============================================================================

import seaborn as sns  # Importa la librería seaborn

for C_value in param_grid['C']:
    modelo_temp = SVC(kernel='linear', C=C_value, random_state=123)
    y_pred_cv = cross_val_predict(modelo_temp, X_train, y_train, cv=cv)

    # Matriz de confusión
    conf_matrix = confusion_matrix(y_train, y_pred_cv)

    # Cálculo de accuracy, precisión y recall
    accuracy = accuracy_score(y_train, y_pred_cv)
    precision = precision_score(y_train, y_pred_cv, average='weighted')
    recall = recall_score(y_train, y_pred_cv, average='weighted')

    # Mostrar los resultados de cada iteración
    print(f"\n--- Evaluación del Modelo SVM con C={C_value} ---")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precisión: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print("Matriz de Confusión:")
    print(conf_matrix)

    # Visualización gráfica de la matriz de confusión
    plt.figure(figsize=(6, 5))  # Configura el tamaño de la figura
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                 xticklabels=['Benigno', 'Maligno'], 
                 yticklabels=['Benigno', 'Maligno'])  # Etiquetas para las clases
    plt.title(f'Matriz de Confusión (C={C_value})')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.show()  # Muestra la gráfica

# Obtener el mejor parámetro C encontrado
# ==============================================================================
best_C = grid_search.best_params_['C']
print(f"\nMejor parámetro C encontrado: {best_C}")

# El mejor modelo encontrado con GridSearchCV
modelo_optimo = grid_search.best_estimator_

# Evaluación final en el conjunto de test
# ==============================================================================
y_pred_test = modelo_optimo.predict(X_test)

# Matriz de confusión para el conjunto de test
conf_matrix_test = confusion_matrix(y_test, y_pred_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test, average='weighted')
recall_test = recall_score(y_test, y_pred_test, average='weighted')

print("\n--- Evaluación final en el conjunto de test ---")
print(f"Accuracy: {accuracy_test * 100:.2f}%")
print(f"Precisión: {precision_test * 100:.2f}%")
print(f"Recall: {recall_test * 100:.2f}%")
print("Matriz de Confusión (test):")
print(conf_matrix_test)

plt.figure(figsize=(6, 5))  # Configura el tamaño de la figura
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', 
             xticklabels=['Benigno', 'Maligno'], 
             yticklabels=['Benigno', 'Maligno'])  # Etiquetas para las clases
plt.title('Matriz de Confusión en el Conjunto de Test')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()  # Muestra la gráfica

# Representación gráfica de los límites de clasificación para el mejor modelo
# ==============================================================================
# Grid de valores para graficar
x = np.linspace(np.min(X_train['mean radius']) - 7, np.max(X_train['mean radius']) + 7, 50)
y = np.linspace(np.min(X_train['mean texture']) - 7, np.max(X_train['mean texture']) + 7, 50)
Y, X_grid = np.meshgrid(x, y)
grid = np.vstack([X_grid.ravel(), Y.ravel()]).T

# Predicción de las clases para los puntos del grid
pred_clases_grid = modelo_optimo.predict(grid)

# Reshape de las predicciones para que coincidan con la forma del grid
Z = pred_clases_grid.reshape(X_grid.shape)

# Graficar resultados
fig, ax = plt.subplots(figsize=(8, 6))

# Usar contourf para rellenar las áreas de las clases con colores de las clases (rojo y azul)
contour = ax.contourf(X_grid, Y, Z, alpha=0.3, cmap='coolwarm')

# Graficar las predicciones del grid con transparencia
ax.scatter(grid[:, 0], grid[:, 1], c=pred_clases_grid, alpha=0.2, cmap='coolwarm')

# Graficar los datos de entrenamiento con las clases reales
scatter = ax.scatter(X_train['mean radius'], X_train['mean texture'], c=y_train, edgecolor='k', cmap='coolwarm')

# Vectores soporte
ax.scatter(
    modelo_optimo.support_vectors_[:, 0],
    modelo_optimo.support_vectors_[:, 1],
    s=200, linewidth=1,
    facecolors='none', edgecolors='black'
)

# Hiperplano de separación
ax.contour(
    X_grid, Y, modelo_optimo.decision_function(grid).reshape(X_grid.shape),
    colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--']
)

# Agregar leyenda que explique los colores
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Benigno', markerfacecolor='blue', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Maligno', markerfacecolor='red', markersize=10),
]

ax.legend(handles=legend_elements, loc='upper right')

ax.set_title("Resultados clasificación SVM lineal (Breast Cancer Dataset)")
ax.set_xlabel("Radio Medio")
ax.set_ylabel("Textura Media")
plt.show()