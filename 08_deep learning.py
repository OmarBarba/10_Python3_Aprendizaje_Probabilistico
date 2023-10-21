import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Cargar el conjunto de datos MNIST
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalizar las imágenes (escala de 0 a 1)
train_images, test_images = train_images / 255.0, test_images / 255.0

# Crear un modelo de red neuronal profunda
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Capa de entrada: aplanar las imágenes
    keras.layers.Dense(128, activation='relu'),   # Capa oculta con 128 neuronas y función de activación ReLU
    keras.layers.Dropout(0.2),                   # Capa de dropout para regularización
    keras.layers.Dense(10, activation='softmax')  # Capa de salida con 10 neuronas para 10 clases y función de activación softmax
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(train_images, train_labels, epochs=5)

# Evaluar el modelo
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Precisión en el conjunto de prueba: {test_acc}')

# Hacer predicciones en un conjunto de prueba
predictions = model.predict(test_images)

# Visualizar una imagen de prueba y su etiqueta predicha
plt.figure()
plt.imshow(test_images[0], cmap=plt.cm.binary)
plt.xlabel(f'Etiqueta real: {test_labels[0]}')
plt.title(f'Etiqueta predicha: {tf.argmax(predictions[0])}')
plt.show()
