import numpy as np

# Datos de ejemplo con dos distribuciones normales
data = np.concatenate([np.random.normal(0, 1, 500), np.random.normal(5, 1, 500)])

# Inicialización de los parámetros: media y desviación estándar de las dos distribuciones
mu1, mu2 = np.random.rand(2)
sigma1, sigma2 = np.random.rand(2)

# Probabilidades iniciales de pertenencia a cada distribución (la suma debe ser 1)
pi1, pi2 = np.random.rand(2)
pi1, pi2 = pi1 / (pi1 + pi2), pi2 / (pi1 + pi2)

# Número de iteraciones EM
num_iteraciones = 100

for _ in range(num_iteraciones):
    # Paso E (Expectation): Calcular las probabilidades de pertenencia a cada distribución
    pdf1 = pi1 * (1 / (sigma1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((data - mu1) / sigma1) ** 2)
    pdf2 = pi2 * (1 / (sigma2 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((data - mu2) / sigma2) ** 2)
    total_prob = pdf1 + pdf2
    gamma1 = pdf1 / total_prob
    gamma2 = pdf2 / total_prob

    # Paso M (Maximization): Actualizar los parámetros
    mu1 = np.sum(gamma1 * data) / np.sum(gamma1)
    mu2 = np.sum(gamma2 * data) / np.sum(gamma2)
    sigma1 = np.sqrt(np.sum(gamma1 * (data - mu1) ** 2) / np.sum(gamma1))
    sigma2 = np.sqrt(np.sum(gamma2 * (data - mu2) ** 2) / np.sum(gamma2))
    pi1 = np.mean(gamma1)
    pi2 = np.mean(gamma2)

print("Parámetros finales:")
print("Media 1:", mu1)
print("Desviación estándar 1:", sigma1)
print("Probabilidad 1:", pi1)
print("Media 2:", mu2)
print("Desviación estándar 2:", sigma2)
print("Probabilidad 2:", pi2)
