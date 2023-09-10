import numpy as np
import matplotlib.pyplot as plt

# --- Entrenar perceptron ---
def funcionActivacion(x):
    return 1 if x >= 0 else -1

def entrenarPerceptron(XEntrenamiento, YEntrenamiento, tazaAprendizaje, epocas, iteracion):
    pesos = np.random.rand(3)
    bias = np.random.rand(1)
    errores = []

    for i in range(epocas):
        errorTotal = 0
        for j in range(len(XEntrenamiento)):
            x = XEntrenamiento[j]
            y = YEntrenamiento[j]

            # Usar funcion de activacion
            salida = funcionActivacion(np.dot(x, pesos) + bias)

            # Calcular error
            error = y - salida

            # Reajustar pesos y bias
            pesos += tazaAprendizaje * error * x
            bias += tazaAprendizaje * error

            errorTotal += np.abs(error)
        errores.append(errorTotal)
    
    plt.plot(range(epocas), errores)
    plt.xlabel("Epoca")
    plt.ylabel("Error")
    plt.title(f"Entrenamiento particion {iteracion + 1}")
    plt.show()

    return pesos, bias

# --- Generar particiones y hacer pruebas ---
def particiones(archivo, particiones):
    datos = np.genfromtxt(archivo, delimiter = ",")
    precisiones = []

    X = datos[:, :3]
    Y = datos[:, 3]

    numeroDatos = X.shape[0]
    numeroCaracteristicas = X.shape[1]

    tamEntrenamiento = int(0.8 * numeroDatos)
    tamPrueba = numeroDatos - tamEntrenamiento

    for i in range(particiones):
        print(f"ARCHIVO {archivo} PARTICION {i + 1}")
        indices = np.random.permutation(numeroDatos)
        X = X[indices]
        Y = Y[indices]

        XEntrenamiento = X[:tamEntrenamiento]
        YEntrenamiento = Y[:tamEntrenamiento]
        XPrueba = X[tamEntrenamiento:]
        YPrueba = Y[tamEntrenamiento:]

        pesos, bias = entrenarPerceptron(XEntrenamiento, YEntrenamiento, 0.1, 100, i)

        predicciones = []
        prediccionesCorrectas = 0
        lenghtX = len(XPrueba)

        for j in range(lenghtX):
            # Sacar predicciones con funcion de activacion y guardarlas
            prediccion = funcionActivacion(np.dot(XPrueba[j], pesos) + bias)
            predicciones.append(prediccion)

            # Contar predicciones correctas
            if prediccion == YPrueba[j]:
                prediccionesCorrectas += 1

        for j in range(lenghtX):
            print(f"Entrada: {XPrueba[j]}, Prediccion: {predicciones[j]}, Real: {YPrueba[j]}")

        precisiones.append((prediccionesCorrectas / lenghtX) * 100)

    
    resultados = [(i + 1, precision) for i, precision in enumerate(precisiones)]

    # Ordenar la lista de resultados por precision en orden descendente (de mejor a peor)
    resultados_ordenados = sorted(resultados, key=lambda x: x[1], reverse=True)

    # Mostrar los resultados
    print("MEJORES PARTICIONES")
    for particion, precision in resultados_ordenados:
        print(f"Particion: {particion}, Precision: {precision}%")

particiones("spheres1d10.csv", 5)
particiones("spheres2d10.csv", 10)
particiones("spheres2d50.csv", 10)
particiones("spheres2d70.csv", 10)
