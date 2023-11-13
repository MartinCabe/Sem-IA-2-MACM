import numpy as np
import tensorflow as tf
from tensorflow.keras.models import clone_model
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score

def particion(x, y):
  # Particionar datos
  num_datos = x.shape[0]
  tam_entrenamiento = int(0.8 * num_datos)
  tam_prueba = num_datos - tam_entrenamiento
  indices = np.random.permutation(num_datos)
  x = x[indices]
  y = y[indices]
  x_entrenamiento = x[:tam_entrenamiento]
  y_entrenamiento = y[:tam_entrenamiento]
  x_prueba = x[tam_entrenamiento:]
  y_prueba = y[tam_entrenamiento:]
  return x_entrenamiento, x_prueba, y_entrenamiento, y_prueba

# Leer archivos
archivo_seguro = np.genfromtxt("AutoInsurSweden.csv", delimiter = ',')
archivo_vino = np.genfromtxt("winequality-white.csv", delimiter = ',')
archivo_diabetes = np.genfromtxt("pima-indians-diabetes.csv", delimiter = ',')

# Separar caracteristicas
x_seguro = archivo_seguro[:, 0]
y_seguro = archivo_seguro[:, 1]
x_vino = archivo_vino[:, :11]
y_vino = archivo_vino[:, 11]
x_diabetes = archivo_diabetes[:, :8]
y_diabetes = archivo_diabetes[:, 8]

# Particionar datos
x_entrenamiento_seguro, x_prueba_seguro, y_entrenamiento_seguro, y_prueba_seguro = particion(x_seguro, y_seguro)
x_entrenamiento_vino, x_prueba_vino, y_entrenamiento_vino, y_prueba_vino = particion(x_vino, y_vino)
x_entrenamiento_diabetes, x_prueba_diabetes, y_entrenamiento_diabetes, y_prueba_diabetes = particion(x_diabetes, y_diabetes)

# Definir modelos
model_seguros = tf.keras.Sequential([
    tf.keras.layers.Input(shape = (1,)),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(1, activation = 'linear')
])
model_vino = tf.keras.Sequential([
    tf.keras.layers.Input(shape = (11,)),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(1, activation = 'linear')
])
model_diabetes = tf.keras.Sequential([
    tf.keras.layers.Input(shape = (8,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

# Compilar modelos
model_seguros.compile(optimizer = 'adam', loss = 'mse', metrics = [tf.keras.metrics.MeanSquaredError()])
model_vino.compile(optimizer = 'adam', loss = 'mse', metrics = [tf.keras.metrics.MeanSquaredError()])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model_diabetes.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

# Obtener predicciones
predictions_seguros = model_seguros.predict(x_prueba_seguro)
predictions_vino = model_vino.predict(x_prueba_vino)
predictions_diabetes = model_diabetes.predict(x_prueba_diabetes)

# ----- Regresion Logistica -----

# Copiar los modelos originales
logistic_model_seguros = clone_model(model_seguros)
logistic_model_vino = clone_model(model_vino)
logistic_model_diabetes = clone_model(model_diabetes)

# Modificar las capas finales
logistic_model_seguros.layers[-1].activation = tf.keras.activations.sigmoid
logistic_model_vino.layers[-1].activation = tf.keras.activations.sigmoid
logistic_model_diabetes.layers[-1].activation = tf.keras.activations.sigmoid

# Compilar los modelos
logistic_model_seguros.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
logistic_model_vino.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
logistic_model_diabetes.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar modelos
logistic_model_seguros.fit(x_entrenamiento_seguro, y_entrenamiento_seguro, epochs=100, batch_size=32, verbose = 0)
logistic_model_vino.fit(x_entrenamiento_vino, y_entrenamiento_vino, epochs=100, batch_size=32, verbose = 0)
logistic_model_diabetes.fit(x_entrenamiento_diabetes, y_entrenamiento_diabetes, epochs=500, verbose = 0)

# Evaluar modelos
logistic_model_seguros.evaluate(x_prueba_seguro, y_prueba_seguro)
logistic_model_vino.evaluate(x_prueba_vino, y_prueba_vino)
logistic_model_diabetes.evaluate(x_prueba_diabetes, y_prueba_diabetes)

# ----- Evaluar con metricas -----

umbral_seguro = np.mean(y_entrenamiento_seguro)
umbral_vino = np.mean(y_entrenamiento_vino)

y_prueba_seguro_binario = np.where(y_prueba_seguro >= umbral_seguro, 1, 0)
y_prueba_vino_binario = np.where(y_prueba_vino >= umbral_vino, 1, 0)

# Generar predicciones
logistic_pred_seguros = (logistic_model_seguros.predict(x_prueba_seguro) > 0.5).astype("int32")
logistic_pred_vino = (logistic_model_vino.predict(x_prueba_vino) > 0.5).astype("int32")
logistic_pred_diabetes = (logistic_model_diabetes.predict(x_prueba_diabetes) > 0.5).astype("int32")

# Accuracy
accuracy_seguros = accuracy_score(y_prueba_seguro_binario, logistic_pred_seguros)
accuracy_vino = accuracy_score(y_prueba_vino_binario, logistic_pred_vino)
accuracy_diabetes = accuracy_score(y_prueba_diabetes, logistic_pred_diabetes)

# Precision
precision_seguros = precision_score(y_prueba_seguro_binario, logistic_pred_seguros)
precision_vino = precision_score(y_prueba_vino_binario, logistic_pred_vino)
precision_diabetes = precision_score(y_prueba_diabetes, logistic_pred_diabetes)

# Sensitivity
sensitivity_seguros = recall_score(y_prueba_seguro_binario, logistic_pred_seguros)
sensitivity_vino = recall_score(y_prueba_vino_binario, logistic_pred_vino)
sensitivity_diabetes = recall_score(y_prueba_diabetes, logistic_pred_diabetes)

# Generar matrices de confusion
conf_seguros = confusion_matrix(y_prueba_seguro_binario, logistic_pred_seguros)
conf_vino = confusion_matrix(y_prueba_vino_binario, logistic_pred_vino)
conf_diabetes = confusion_matrix(y_prueba_diabetes, logistic_pred_diabetes)

# Specifity
specificity_seguros = conf_seguros[0, 0] / (conf_seguros[0, 0] + conf_seguros[0, 1])
specificity_vino = conf_vino[0, 0] / (conf_vino[0, 0] + conf_vino[0, 1])
specificity_diabetes = conf_diabetes[0, 0] / (conf_diabetes[0, 0] + conf_diabetes[0, 1])

# F1
f1_seguros = f1_score(y_prueba_seguro_binario, logistic_pred_seguros)
f1_vino = f1_score(y_prueba_vino_binario, logistic_pred_vino)
f1_diabetes = f1_score(y_prueba_diabetes, logistic_pred_diabetes)

# Imprimir resultados
print("----- REGRESION LOGISTICA -----")

print("Modelo de Seguros:")
print(f"Accuracy: {accuracy_seguros}")
print(f"Precision: {precision_seguros}")
print(f"Sensitivity: {sensitivity_seguros}")
print(f"Specificity: {specificity_seguros}")
print(f"F1 Score: {f1_seguros}")
print()

print("Modelo de Vino:")
print(f"Accuracy: {accuracy_vino}")
print(f"Precision: {precision_vino}")
print(f"Sensitivity: {sensitivity_vino}")
print(f"Specificity: {specificity_vino}")
print(f"F1 Score: {f1_vino}")
print()

print("Modelo de Diabetes:")
print(f"Accuracy: {accuracy_diabetes}")
print(f"Precision: {precision_diabetes}")
print(f"Sensitivity: {sensitivity_diabetes}")
print(f"Specificity: {specificity_diabetes}")
print(f"F1 Score: {f1_diabetes}")
print()

# ----- K-Vecinos Cercanos -----

def mostrar_predicciones(predicciones, reales, modelo):
  print(f"\nModelo: {modelo}")
  for i in range(10):
    print(f"Salida real: {reales[i]}, Prediccion: {predicciones[i]}")

k = 5  # Numero de vecinos
knn_seguros = KNeighborsRegressor(n_neighbors=k)
knn_seguros.fit(predictions_seguros, y_prueba_seguro)

knn_vino = KNeighborsRegressor(n_neighbors=k)
knn_vino.fit(predictions_vino, y_prueba_vino)

knn_diabetes = KNeighborsRegressor(n_neighbors=k)
knn_diabetes.fit(predictions_diabetes, y_prueba_diabetes)

# Realiza predicciones
knn_predictions_seguros = knn_seguros.predict(predictions_seguros)
knn_predictions_vino = knn_vino.predict(predictions_vino)
knn_predictions_diabetes = knn_diabetes.predict(predictions_diabetes)

# Mostrar predicciones
mostrar_predicciones(knn_predictions_seguros, predictions_seguros, "Seguros")
mostrar_predicciones(knn_predictions_vino, predictions_vino, "Vino")
mostrar_predicciones(knn_predictions_diabetes, predictions_diabetes, "Diabetes")

# ----- Evaluar con metricas -----

umbral_diabetes = np.mean(y_entrenamiento_diabetes)

knn_predictions_seguros_binario = np.where(knn_predictions_seguros >= umbral_seguro, 1, 0)
knn_predictions_vino_binario = np.where(knn_predictions_vino >= umbral_vino, 1, 0)
knn_predictions_diabetes_binario = np.where(knn_predictions_diabetes >= umbral_seguro, 1, 0)

# Accuracy
accuracy_seguros = accuracy_score(y_prueba_seguro_binario, knn_predictions_seguros_binario)
accuracy_vino = accuracy_score(y_prueba_vino_binario, knn_predictions_vino_binario)
accuracy_diabetes = accuracy_score(y_prueba_diabetes, knn_predictions_diabetes_binario)

# Precision
precision_seguros = precision_score(y_prueba_seguro_binario, knn_predictions_seguros_binario)
precision_vino = precision_score(y_prueba_vino_binario, knn_predictions_vino_binario)
precision_diabetes = precision_score(y_prueba_diabetes, knn_predictions_diabetes_binario)

# Sensitivity
sensitivity_seguros = recall_score(y_prueba_seguro_binario, knn_predictions_seguros_binario)
sensitivity_vino = recall_score(y_prueba_vino_binario, knn_predictions_vino_binario)
sensitivity_diabetes = recall_score(y_prueba_diabetes, knn_predictions_diabetes_binario)

# Generar matrices de confusion
conf_seguros = confusion_matrix(y_prueba_seguro_binario, knn_predictions_seguros_binario)
conf_vino = confusion_matrix(y_prueba_vino_binario, knn_predictions_vino_binario)
conf_diabetes = confusion_matrix(y_prueba_diabetes, knn_predictions_diabetes_binario)

# Specifity
specificity_seguros = conf_seguros[0, 0] / (conf_seguros[0, 0] + conf_seguros[0, 1])
specificity_vino = conf_vino[0, 0] / (conf_vino[0, 0] + conf_vino[0, 1])
specificity_diabetes = conf_diabetes[0, 0] / (conf_diabetes[0, 0] + conf_diabetes[0, 1])

# F1
f1_seguros = f1_score(y_prueba_seguro_binario, knn_predictions_seguros_binario)
f1_vino = f1_score(y_prueba_vino_binario, knn_predictions_vino_binario)
f1_diabetes = f1_score(y_prueba_diabetes, knn_predictions_diabetes_binario)

# Imprimir resultados
print("----- K VECINOS CERCANOS -----")

print("Modelo de Seguros:")
print(f"Accuracy: {accuracy_seguros}")
print(f"Precision: {precision_seguros}")
print(f"Sensitivity: {sensitivity_seguros}")
print(f"Specificity: {specificity_seguros}")
print(f"F1 Score: {f1_seguros}")
print()

print("Modelo de Vino:")
print(f"Accuracy: {accuracy_vino}")
print(f"Precision: {precision_vino}")
print(f"Sensitivity: {sensitivity_vino}")
print(f"Specificity: {specificity_vino}")
print(f"F1 Score: {f1_vino}")
print()

print("Modelo de Diabetes:")
print(f"Accuracy: {accuracy_diabetes}")
print(f"Precision: {precision_diabetes}")
print(f"Sensitivity: {sensitivity_diabetes}")
print(f"Specificity: {specificity_diabetes}")
print(f"F1 Score: {f1_diabetes}")
print()

# ----- Maquinas Vector Soporte -----

# Cambiar a vectores 2D
x_entrenamiento_seguro_2d = x_entrenamiento_seguro.reshape(-1, 1)
x_prueba_seguro_2d = x_prueba_seguro.reshape(-1, 1)

# Definir modelos
svm_seguro = svm.SVR(kernel='linear')
svm_vino  = svm.SVR(kernel='linear')
svm_diabetes = svm.SVC(probability=True)

# Entrenar modelos
svm_seguro.fit(x_entrenamiento_seguro_2d, y_entrenamiento_seguro)
svm_vino.fit(x_entrenamiento_vino, y_entrenamiento_vino)
svm_diabetes.fit(x_entrenamiento_diabetes, y_entrenamiento_diabetes)

# Evaluar los modelos
score_seguro = svm_seguro.score(x_prueba_seguro_2d, y_prueba_seguro)
score_vino = svm_vino.score(x_prueba_vino, y_prueba_vino)
score_diabetes = svm_diabetes.score(x_prueba_diabetes, y_prueba_diabetes)

print("SVM Score Seguro:", score_seguro)
print("SVM Score Vino:", score_vino)
print("SVM Score Diabetes:", score_diabetes)

svm_predictions_seguro = svm_seguro.predict(x_prueba_seguro_2d)
svm_predictions_vino = svm_vino.predict(x_prueba_vino)
svm_predictions_diabetes = svm_diabetes.predict(x_prueba_diabetes)

svm_predictions_seguros_binario = np.where(svm_predictions_seguro >= umbral_seguro, 1, 0)
svm_predictions_vino_binario = np.where(svm_predictions_vino >= umbral_vino, 1, 0)
svm_predictions_diabetes_binario = np.where(svm_predictions_diabetes >= umbral_seguro, 1, 0)

# Accuracy
accuracy_seguros = accuracy_score(y_prueba_seguro_binario, svm_predictions_seguros_binario)
accuracy_vino = accuracy_score(y_prueba_vino_binario, svm_predictions_vino_binario)
accuracy_diabetes = accuracy_score(y_prueba_diabetes, svm_predictions_diabetes_binario)

# Precision
precision_seguros = precision_score(y_prueba_seguro_binario, svm_predictions_seguros_binario)
precision_vino = precision_score(y_prueba_vino_binario, svm_predictions_vino_binario)
precision_diabetes = precision_score(y_prueba_diabetes, svm_predictions_diabetes_binario)

# Sensitivity
sensitivity_seguros = recall_score(y_prueba_seguro_binario, svm_predictions_seguros_binario)
sensitivity_vino = recall_score(y_prueba_vino_binario, svm_predictions_vino_binario)
sensitivity_diabetes = recall_score(y_prueba_diabetes, svm_predictions_diabetes_binario)

# Generar matrices de confusion
conf_seguros = confusion_matrix(y_prueba_seguro_binario, svm_predictions_seguros_binario)
conf_vino = confusion_matrix(y_prueba_vino_binario, svm_predictions_vino_binario)
conf_diabetes = confusion_matrix(y_prueba_diabetes, svm_predictions_diabetes_binario)

# Specifity
specificity_seguros = conf_seguros[0, 0] / (conf_seguros[0, 0] + conf_seguros[0, 1])
specificity_vino = conf_vino[0, 0] / (conf_vino[0, 0] + conf_vino[0, 1])
specificity_diabetes = conf_diabetes[0, 0] / (conf_diabetes[0, 0] + conf_diabetes[0, 1])

# F1
f1_seguros = f1_score(y_prueba_seguro_binario, svm_predictions_seguros_binario)
f1_vino = f1_score(y_prueba_vino_binario, svm_predictions_vino_binario)
f1_diabetes = f1_score(y_prueba_diabetes, svm_predictions_diabetes_binario)

# Imprimir resultados
print("----- MAQUINAS VECTOR SOPORTE -----")

print("Modelo de Seguros:")
print(f"Accuracy: {accuracy_seguros}")
print(f"Precision: {precision_seguros}")
print(f"Sensitivity: {sensitivity_seguros}")
print(f"Specificity: {specificity_seguros}")
print(f"F1 Score: {f1_seguros}")
print()

print("Modelo de Vino:")
print(f"Accuracy: {accuracy_vino}")
print(f"Precision: {precision_vino}")
print(f"Sensitivity: {sensitivity_vino}")
print(f"Specificity: {specificity_vino}")
print(f"F1 Score: {f1_vino}")
print()

print("Modelo de Diabetes:")
print(f"Accuracy: {accuracy_diabetes}")
print(f"Precision: {precision_diabetes}")
print(f"Sensitivity: {sensitivity_diabetes}")
print(f"Specificity: {specificity_diabetes}")
print(f"F1 Score: {f1_diabetes}")
print()

# ----- Naive Bayes -----

# Para este metodo, es necesario cambiar a labels binarios o no sera posible
# aplicarlo para los primeros 2 datasets

# Calcular un umbral para definir el nuevo valor binario de los labels a partir
# del promedio
umbral_seguro = np.mean(y_entrenamiento_seguro)
umbral_vino = np.mean(y_entrenamiento_vino)

y_entrenamiento_seguro_binario = np.where(y_entrenamiento_seguro >= umbral_seguro, 1, 0)
y_entrenamiento_vino_binario = np.where(y_entrenamiento_vino >= umbral_vino, 1, 0)
y_prueba_seguro_binario = np.where(y_prueba_seguro >= umbral_seguro, 1, 0)
y_prueba_vino_binario = np.where(y_prueba_vino >= umbral_vino, 1, 0)

# Definir modelos
nb_seguro = GaussianNB()
nb_vino = GaussianNB()
nb_diabetes = BernoulliNB()

# Entrenar modelos
nb_seguro.fit(x_entrenamiento_seguro_2d, y_entrenamiento_seguro_binario)
nb_vino.fit(x_entrenamiento_vino, y_entrenamiento_vino_binario)
nb_diabetes.fit(x_entrenamiento_diabetes, y_entrenamiento_diabetes)

# Evaluar modelos
nb_predictions_seguro = nb_seguro.predict(x_prueba_seguro_2d)
nb_predictions_vino = nb_vino.predict(x_prueba_vino)
nb_predictions_diabetes = nb_diabetes.predict(x_prueba_diabetes)

nb_accuracy_seguro = accuracy_score(y_prueba_seguro_binario, nb_predictions_seguro)
nb_accuracy_vino = accuracy_score(y_prueba_vino_binario, nb_predictions_vino)
nb_accuracy_diabetes = accuracy_score(y_prueba_diabetes, nb_predictions_diabetes)

print("Naive Bayes Accuracy Seguro:", nb_accuracy_seguro)
print("Naive Bayes Accuracy Vino:", nb_accuracy_vino)
print("Naive Bayes Accuracy Diabetes:", nb_accuracy_diabetes)

nb_predictions_seguros_binario = np.where(nb_predictions_seguro >= umbral_seguro, 1, 0)
nb_predictions_vino_binario = np.where(nb_predictions_vino >= umbral_vino, 1, 0)
nb_predictions_diabetes_binario = np.where(nb_predictions_diabetes >= umbral_seguro, 1, 0)

# Accuracy
accuracy_seguros = accuracy_score(y_prueba_seguro_binario, nb_predictions_seguros_binario)
accuracy_vino = accuracy_score(y_prueba_vino_binario, nb_predictions_vino_binario)
accuracy_diabetes = accuracy_score(y_prueba_diabetes, nb_predictions_diabetes_binario)

# Precision
precision_seguros = precision_score(y_prueba_seguro_binario, nb_predictions_seguros_binario)
precision_vino = precision_score(y_prueba_vino_binario, nb_predictions_vino_binario)
precision_diabetes = precision_score(y_prueba_diabetes, nb_predictions_diabetes_binario)

# Sensitivity
sensitivity_seguros = recall_score(y_prueba_seguro_binario, nb_predictions_seguros_binario)
sensitivity_vino = recall_score(y_prueba_vino_binario, nb_predictions_vino_binario)
sensitivity_diabetes = recall_score(y_prueba_diabetes, nb_predictions_diabetes_binario)

# Generar matrices de confusion
conf_seguros = confusion_matrix(y_prueba_seguro_binario, nb_predictions_seguros_binario)
conf_vino = confusion_matrix(y_prueba_vino_binario, nb_predictions_vino_binario)
conf_diabetes = confusion_matrix(y_prueba_diabetes, nb_predictions_diabetes_binario)

# Specifity
specificity_seguros = conf_seguros[0, 0] / (conf_seguros[0, 0] + conf_seguros[0, 1])
specificity_vino = conf_vino[0, 0] / (conf_vino[0, 0] + conf_vino[0, 1])
specificity_diabetes = conf_diabetes[0, 0] / (conf_diabetes[0, 0] + conf_diabetes[0, 1])

# F1
f1_seguros = f1_score(y_prueba_seguro_binario, nb_predictions_seguros_binario)
f1_vino = f1_score(y_prueba_vino_binario, nb_predictions_vino_binario)
f1_diabetes = f1_score(y_prueba_diabetes, nb_predictions_diabetes_binario)

# Imprimir resultados
print("----- NAIVE BAYES -----")

print("Modelo de Seguros:")
print(f"Accuracy: {accuracy_seguros}")
print(f"Precision: {precision_seguros}")
print(f"Sensitivity: {sensitivity_seguros}")
print(f"Specificity: {specificity_seguros}")
print(f"F1 Score: {f1_seguros}")
print()

print("Modelo de Vino:")
print(f"Accuracy: {accuracy_vino}")
print(f"Precision: {precision_vino}")
print(f"Sensitivity: {sensitivity_vino}")
print(f"Specificity: {specificity_vino}")
print(f"F1 Score: {f1_vino}")
print()

print("Modelo de Diabetes:")
print(f"Accuracy: {accuracy_diabetes}")
print(f"Precision: {precision_diabetes}")
print(f"Sensitivity: {sensitivity_diabetes}")
print(f"Specificity: {specificity_diabetes}")
print(f"F1 Score: {f1_diabetes}")
print()