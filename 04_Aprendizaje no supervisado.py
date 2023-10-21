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


#########################################################################################################
#########################################SIN LIBRERIAS###################################################
#########################################################################################################

import numpy as np
import random
import matplotlib.pyplot as plt

# Datos de ejemplo
data = np.array([[2, 3], [2, 5], [8, 1], [5, 8], [7, 2], [6, 6]])

# Número de grupos (clusters) deseados
k = 2

# Inicialización aleatoria de centroides
centroids = [data[i] for i in random.sample(range(len(data)), k)]

# Número máximo de iteraciones
max_iter = 100

for iter in range(max_iter):
    # Inicializar clusters vacíos
    clusters = [[] for _ in range(k)]

    # Asignar cada punto de datos al cluster más cercano
    for point in data:
        distances = [np.linalg.norm(point - centroid) for centroid in centroids]
        cluster_id = np.argmin(distances)
        clusters[cluster_id].append(point)

    # Calcular nuevos centroides como el promedio de los puntos en cada cluster
    new_centroids = [np.mean(cluster, axis=0) for cluster in clusters]

    # Comprobar si los centroides han convergido
    if np.array_equal(centroids, new_centroids):
        break

    # Actualizar centroides
    centroids = new_centroids

print("Centroides finales:")
for i, centroid in enumerate(centroids):
    print(f"Cluster {i + 1}: {centroid}")

print("Puntos por cluster:")
for i, cluster in enumerate(clusters):
    print(f"Cluster {i + 1}: {cluster}")

# Graficar los resultados
colors = ['b', 'g', 'r', 'c', 'm', 'y']
for i, cluster in enumerate(clusters):
    points = np.array(cluster)
    plt.scatter(points[:, 0], points[:, 1], c=colors[i], label=f'Cluster {i + 1}')

centroid_points = np.array(centroids)
plt.scatter(centroid_points[:, 0], centroid_points[:, 1], marker='x', s=100, c='k', label='Centroides')

plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.title('Resultados del Agrupamiento K-Means')
plt.legend()
plt.grid(True)
plt.show()
