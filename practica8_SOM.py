import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors

# Cargar el dataset de pingüinos desde seaborn
data = sns.load_dataset('penguins')

# Preprocesar los datos: eliminar filas con valores nulos y seleccionar características
data = data.dropna()
features = data[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
labels = data['species']

# Normalizar las características
normalized_features = (features - features.mean()) / features.std()

# Parámetros para tres tamaños de SOM
som_sizes = [(7, 7), (10, 10), (13, 13)]  # Tamaños de SOM a probar
input_dim = normalized_features.shape[1]  # Dimensiones de entrada

# Definir color para cada especie
color_map = {'Adelie': 'red', 'Chinstrap': 'green', 'Gentoo': 'blue'}

for som_x, som_y in som_sizes:
    # Inicializar los pesos de cada neurona en la cuadrícula con valores aleatorios entre 0 y 1
    weights = np.random.rand(som_x, som_y, input_dim)

    # Función para encontrar la neurona ganadora (BMU)
    def find_bmu(input_vector, weights):
        distances = np.sqrt(((weights - input_vector) ** 2).sum(axis=2))
        bmu_index = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
        return bmu_index

    # Función para actualizar los pesos
    def update_weights(input_vector, weights, bmu_index, t, max_iter, init_learning_rate=0.5, init_radius=3.0):
        learning_rate = init_learning_rate * (1 - t / max_iter)
        radius = init_radius * (1 - t / max_iter)
        
        for i in range(som_x):
            for j in range(som_y):
                dist_to_bmu = np.sqrt((i - bmu_index[0]) ** 2 + (j - bmu_index[1]) ** 2)
                if dist_to_bmu <= radius:
                    influence = np.exp(-dist_to_bmu ** 2 / (2 * (radius ** 2)))
                    weights[i, j] += learning_rate * influence * (input_vector - weights[i, j])

    # Entrenamiento del SOM
    max_iter = 1000
    for t in range(max_iter):
        input_vector = normalized_features.iloc[np.random.randint(0, len(normalized_features))].values
        bmu_index = find_bmu(input_vector, weights)
        update_weights(input_vector, weights, bmu_index, t, max_iter)

    print(f"Entrenamiento completo para SOM de tamaño {som_x}x{som_y}.")

    # Crear un mapa de colores basado en la frecuencia de cada especie en cada neurona
    frequency_map = np.zeros((som_x, som_y))
    species_map = np.empty((som_x, som_y), dtype=object)
    species_frequency = {species: np.zeros((som_x, som_y)) for species in color_map.keys()}

    for idx in range(len(normalized_features)):
        input_vector = normalized_features.iloc[idx].values
        bmu_index = find_bmu(input_vector, weights)
        if species_map[bmu_index] is None:
            species_map[bmu_index] = labels.iloc[idx]
        else:
            species_map[bmu_index] += f", {labels.iloc[idx]}"
        frequency_map[bmu_index] += 1
        species_frequency[labels.iloc[idx]][bmu_index] += 1

    # Visualización del SOM
    fig, ax = plt.subplots(figsize=(10, 10))
    for i in range(som_x):
        for j in range(som_y):
            if frequency_map[i][j] > 0:
                species_count = {species: 0 for species in color_map.keys()}
                for idx in range(len(normalized_features)):
                    input_vector = normalized_features.iloc[idx].values
                    if find_bmu(input_vector, weights) == (i, j):
                        species_count[labels.iloc[idx]] += 1
                predominant_species = max(species_count.items(), key=lambda x: x[1])[0]
                color_intensity = min(frequency_map[i][j] / frequency_map.max(), 1)
                ax.fill_between([i, i + 1], [j, j], [j + 1, j + 1], 
                                color=color_map[predominant_species], alpha=color_intensity,
                                edgecolor='k')
            else:
                ax.fill_between([i, i + 1], [j, j], [j + 1, j + 1], color='white', edgecolor='k')

    # Añadir simbología para cada especie
    for species, color in color_map.items():
        ax.plot([], [], color=color, label=species, linewidth=10)
    ax.legend(loc='upper right', title="Especie", bbox_to_anchor=(1.12, 1))

    # Crear barras de color para cada especie a la izquierda de la cuadrícula
    cbar_ax_adelie = fig.add_axes([0.1, 0.15, 0.02, 0.7])
    cbar_ax_chinstrap = fig.add_axes([0.15, 0.15, 0.02, 0.7])
    cbar_ax_gentoo = fig.add_axes([0.2, 0.15, 0.02, 0.7])
    norm = colors.Normalize(vmin=0, vmax=frequency_map.max())
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='Reds'), cax=cbar_ax_adelie, label='Adelie')
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='Greens'), cax=cbar_ax_chinstrap, label='Chinstrap')
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='Blues'), cax=cbar_ax_gentoo, label='Gentoo')

    ax.set_title(f"Mapa Autoorganizado de Pingüinos (SOM de {som_x}x{som_y})")
    ax.axis('off')
    plt.subplots_adjust(left=0.3)
    plt.show()
