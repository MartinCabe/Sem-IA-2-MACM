import numpy as np
import matplotlib.pyplot as plt

archivoEntrenamiento = "XOR_trn.csv"
archivoPrueba = "XOR_tst.csv"

# Guardar datos de entrenamiento
datosEntrenamiento = np.genfromtxt(archivoEntrenamiento, delimiter=",")
x_entrenamiento = datosEntrenamiento[:, :2]
y_entrenamiento = datosEntrenamiento[:, 2]

# Calcular pesos y bias aleatorios
pesos = np.random.rand(2)
bias = np.random.rand(1)

tazaAprendizaje = 0.5

def funcionActivacion(x):
    return 1 if x >= 0 else -1

epocas = 100
errores = []

# Entrenamiento del perceptron
for i in range(epocas):
    errorTotal = 0
    for j in range(len(x_entrenamiento)):
        x = x_entrenamiento[j]
        y = y_entrenamiento[j]

        # Usar funcion de activacion
        salida = funcionActivacion(np.dot(x, pesos) + bias)

        # Calcular error
        error = y - salida

        # Reajustar pesos y bias
        pesos += tazaAprendizaje * error * x
        bias += tazaAprendizaje * error

        errorTotal += np.abs(error)
    errores.append(errorTotal)

plt.figure(1)

plt.plot(range(epocas), errores)
plt.xlabel("Epoca")
plt.ylabel("Error")
plt.title("Entrenamiento")

# Crear linea de separacion
xLinea = np.linspace(-1.1, 1.1, 200)
yLinea = (-pesos[0] * xLinea - bias) / pesos[1]

plt.figure(2)

plt.plot(xLinea, yLinea, 'r--', label='Línea de Separación')
plt.scatter(x_entrenamiento[:, 0], x_entrenamiento[:, 1], c=y_entrenamiento, cmap='viridis', s=100)
plt.legend()
plt.title("Separacion despues del entrenamiento")

plt.show()

# Guardar datos de prueba
datosPrueba = np.genfromtxt(archivoPrueba, delimiter=",")
x_prueba = datosPrueba[:, :2]
y_real = datosPrueba[:, 2]

salidaPrueba = []
prediccionesCorrectas = 0
lenghtX = len(x_prueba)

for i in range(lenghtX):
    # Sacar predicciones con funcion de activacion y guardarlas
    prediccion = funcionActivacion(np.dot(x_prueba[i], pesos) + bias)
    salidaPrueba.append(prediccion)

    # Contar predicciones correctas
    if prediccion == y_real[i]:
        prediccionesCorrectas += 1

for i in range(lenghtX):
    print("Entrada: "+str(x_prueba[i])+", Prediccion: "+str(salidaPrueba[i]))

print("Precision: "+str((prediccionesCorrectas/lenghtX) * 100)+"%")
