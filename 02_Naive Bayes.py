import re
from collections import defaultdict

# Datos de ejemplo: correos electrónicos y sus etiquetas (spam o no spam)
correos = [
    ("Oferta especial: gane un millón de dólares", "spam"),
    ("Reunión de la junta directiva a las 3 PM", "no_spam"),
    ("Ganadores del concurso: reclame su premio", "spam"),
    ("Actualización importante de su cuenta bancaria", "spam"),
    ("Confirmación de la reserva de vuelo", "no_spam"),
    ("Descuento del 20% en su próxima compra", "no_spam")
]

# Función para preprocesar el texto (tokenización simple)
def preprocess(text):
    return re.findall(r'\w+', text.lower())

# Construir un diccionario de frecuencia de palabras por clase
word_count = defaultdict(lambda: defaultdict(int))
class_count = defaultdict(int)

for texto, etiqueta in correos:
    for palabra in preprocess(texto):
        word_count[etiqueta][palabra] += 1
        class_count[etiqueta] += 1

# Función para calcular la probabilidad de una palabra en una clase
def word_probability(word, etiqueta, alpha=1.0):
    return (word_count[etiqueta][word] + alpha) / (class_count[etiqueta] + alpha * len(word_count[etiqueta]))

# Función para predecir la etiqueta de un texto
def predict(text):
    text_words = preprocess(text)
    best_label = None
    best_score = float('-inf')

    for etiqueta in class_count.keys():
        log_prob = 0.0

        for palabra in text_words:
            log_prob += -1.0 * word_probability(palabra, etiqueta)  # Usar log para evitar underflow

        log_prob += class_count[etiqueta] / sum(class_count.values())  # Probabilidad a priori
        if log_prob > best_score:
            best_label = etiqueta
            best_score = log_prob

    return best_label

# Realizar predicciones
for texto, etiqueta in correos:
    pred = predict(texto)
    print(f"Texto: {texto}\nEtiqueta real: {etiqueta}\nEtiqueta predicha: {pred}\n")
