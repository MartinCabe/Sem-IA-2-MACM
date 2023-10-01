import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

archivoEntrenamiento = np.genfromtxt("concentlite.csv", delimiter = ',')
x_entrenamiento = archivoEntrenamiento[:, :2]
y_entrenamiento = archivoEntrenamiento[:, 2]

archivoPrueba = np.genfromtxt("concentlite_tst.csv", delimiter = ',')
x_prueba = archivoPrueba[:, :2]
y_prueba = archivoPrueba[:, 2]

# Declarar modelo, capas y funciones de activacion
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation = 'relu', input_shape = (2,)), # Capa de entrada

    # Capas ocultas
    tf.keras.layers.Dense(8, activation = 'relu'),
    tf.keras.layers.Dense(8, activation = 'relu'),

    tf.keras.layers.Dense(1, activation = 'sigmoid') # Capa de salida
])

# Declarar algoritmos de optimizacion, perdida y taza de aprendizaje
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

# Entrenar modelo
model.fit(x_entrenamiento, y_entrenamiento, epochs = 1000, verbose = 0)

# Evaluar modelo
loss, accuracy = model.evaluate(x_prueba, y_prueba)
print(f"loss: {loss}, Accuracy: {accuracy}")

# Probar modelo
predictions = model.predict(x_prueba)
colores = ['black' if pred >= 0.5 else 'red' for pred in predictions] # Colores para la grafica

# Resultados numericos
print("Predicciones:")
for i in range(len(predictions)):
    print(f"Entrada: {x_prueba[i]}, Salida real: {y_prueba[i]}, Prediccion: {predictions[i][0]}")

# Resultados graficos
plt.scatter(x_prueba[:, 0],x_prueba[:, 1],c = colores, alpha=0.5)
plt.xlabel('Caracteristica 1')
plt.ylabel('Caracteristica 2')
plt.title('Clasificacion')
plt.show()