import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut, LeavePOut

# Cargar datos
archivo = np.genfromtxt("irisbin.csv", delimiter = ',')

x = archivo[:, :4]
y = archivo[:, 4:]

# Hacer particion
numero_datos = x.shape[0]
tam_entrenamiento = int(0.8 * numero_datos)
indices_permutacion = np.random.permutation(numero_datos)

x = x[indices_permutacion]
y = y[indices_permutacion]

x_entrenamiento = x[:tam_entrenamiento]
y_entrenamiento = y[:tam_entrenamiento]

x_prueba = x[tam_entrenamiento:]
y_prueba = y[tam_entrenamiento:]

# Declarar modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Input(shape = (4,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(16, activation = 'relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(8, activation = 'relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(8, activation = 'relu'),
    tf.keras.layers.Dense(3, activation = 'softmax')
])

modelo.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Entrenar modelo
modelo.fit(x_entrenamiento, y_entrenamiento, epochs = 1000, verbose = 0)

def evaluar_modelo(x, y, modelo):
    y_verdadera = np.argmax(y, axis = 1)
    y_prediccion = np.argmax(modelo.predict(x), axis=1)
    acc = accuracy_score(y_verdadera, y_prediccion)
    return acc

# Validar con Leave One Out
loo = LeaveOneOut()
loo_accuracies = []
for indice_entrenamiento, indice_prueba in loo.split(x):
    x_train, x_test = x[indice_entrenamiento], x[indice_prueba]
    y_train, y_test = y[indice_entrenamiento], y[indice_prueba]
    model = tf.keras.models.clone_model(modelo)
    acc = evaluar_modelo(x_test, y_test, model)
    loo_accuracies.append(acc)

# Validar con Leave P Out
lpo = LeavePOut(p = 2)
lpo_accuracies = []
for indice_entrenamiento, indice_prueba in lpo.split(x):
    x_train, x_test = x[indice_entrenamiento], x[indice_prueba]
    y_train, y_test = y[indice_entrenamiento], y[indice_prueba]
    model = tf.keras.models.clone_model(modelo)
    acc = evaluar_modelo(x_test, y_test, model)
    lpo_accuracies.append(acc)

# Calcular estadisticas
loo_mean_acc = np.mean(loo_accuracies)
loo_std_acc = np.std(loo_accuracies)
lpo_mean_acc = np.mean(lpo_accuracies)
lpo_std_acc = np.std(lpo_accuracies)

print(f'Leave-One-Out Mean Accuracy: {loo_mean_acc}')
print(f'Leave-One-Out Std Deviation: {loo_std_acc}')
print(f'Leave-P-Out Mean Accuracy: {lpo_mean_acc}')
print(f'Leave-P-Out Std Deviation: {lpo_std_acc}')