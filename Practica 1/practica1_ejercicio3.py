import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Particionar datos, 80-20
archivo = np.genfromtxt("concentlite.csv", delimiter = ',')

x = archivo[:, :2]
y = archivo[:, 2]

numero_datos = x.shape[0]
numero_caracteristicas = x.shape[1]

tam_entrenamiento = int(0.8 * numero_datos)
tam_prueba = numero_datos - tam_entrenamiento

indices = np.random.permutation(numero_datos)

x = x[indices]
y = y[indices]

x_entrenamiento = x[:tam_entrenamiento]
y_entrenamiento = y[:tam_entrenamiento]

x_prueba = x[tam_entrenamiento:]
y_prueba = y[tam_entrenamiento:]

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