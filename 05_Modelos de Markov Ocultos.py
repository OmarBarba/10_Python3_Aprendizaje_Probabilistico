import numpy as np
from hmmlearn import hmm

# Definición del modelo HMM
model = hmm.GaussianHMM(n_components=2, covariance_type="full")

# Parámetros del modelo
model.startprob_ = np.array([0.5, 0.5])  # Probabilidades iniciales de los estados ocultos
model.transmat_ = np.array([[0.7, 0.3], [0.3, 0.7]])  # Matriz de transición de estados ocultos
model.means_ = np.array([[0.0, 0.0], [3.0, 3.0]])  # Medias de las distribuciones gaussianas
model.covars_ = np.tile(np.identity(2), (2, 1, 1))  # Matrices de covarianza

# Generación de secuencias de observaciones
num_samples = 100
hidden_states, observations = model.sample(num_samples)

# Entrenamiento del modelo HMM
model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100)
model.fit(observations)

# Predicción del estado oculto y la probabilidad de la secuencia observada
sequence = np.random.randn(num_samples, 2)  # Secuencia de observaciones de ejemplo
predicted_states = model.predict(sequence)
log_prob = model.score(sequence)

# Resultados
print("Secuencia de estados ocultos:")
print(hidden_states)
print("Secuencia de observaciones:")
print(observations)
print("Predicción de estados ocultos:")
print(predicted_states)
print("Log Probabilidad de la secuencia observada:", log_prob)
