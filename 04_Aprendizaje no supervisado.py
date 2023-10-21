import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generar datos de ejemplo
data, labels = make_blobs(n_samples=300, centers=3, random_state=42)

# Crear y entrenar el modelo K-Means
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)

# Obtener las etiquetas de clúster asignadas a cada punto de datos
cluster_labels = kmeans.labels_

# Obtener las coordenadas de los centroides
centroids = kmeans.cluster_centers_

# Visualizar los resultados
plt.figure(figsize=(8, 6))

# Dibujar los puntos de datos agrupados por clúster
for i in range(3):
    plt.scatter(data[cluster_labels == i, 0], data[cluster_labels == i, 1], label=f'Cluster {i + 1}')

# Dibujar los centroides
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200, label='Centroides')

plt.title('Agrupamiento K-Means')
plt.legend()
plt.show()
