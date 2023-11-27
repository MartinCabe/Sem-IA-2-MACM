import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

def accuracy(y, predictions):
  print(f"Accuracy: {accuracy_score(y, predictions)}")

def precision(y, predictions):
  print(f"Precision: {precision_score(y, predictions, average='micro')}")

def sensitivity(y, predictions):
  print(f"Sensitivity: {recall_score(y, predictions, average='micro')}")

def specifity(y, predictions):
  confusion = confusion_matrix(y, predictions)
  print(f"Specifity: {confusion[0, 0] / (confusion[0, 0] + confusion[0, 1])}")

def f1(y, predictions):
  print(f"F1: {f1_score(y, predictions, average = 'micro')}")

# Leer archivo
archivo = np.genfromtxt("zoo.csv", delimiter = ',')

# Separar caracteristicas
x = archivo[:, 1:-1]
y = archivo[:, 17] - 1 # Cambiar a un rango de 0 a 6
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Particionar datos
x_entrenamiento, x_prueba, y_entrenamiento, y_prueba = train_test_split(x, y, test_size=0.2, random_state=42)

# Declarar modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(x_entrenamiento.shape[1],)),
    tf.keras.layers.Dense(7, activation='softmax')
])

# Compilar modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ----- REGRESION LOGISTICA -----
print("----- REGRESION LOGISTICA -----")

logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(x_entrenamiento, y_entrenamiento)

logistic_predictions = logistic_model.predict(x_prueba)

accuracy(y_prueba, logistic_predictions)
precision(y_prueba, logistic_predictions)
sensitivity(y_prueba, logistic_predictions)
specifity(y_prueba, logistic_predictions)
f1(y_prueba, logistic_predictions)

# ----- K VECINOS CERCANOS -----
print("----- K VECINOS CERCANOS -----")

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(x_entrenamiento, y_entrenamiento)

knn_predictions = knn_model.predict(x_prueba)

accuracy(y_prueba, knn_predictions)
precision(y_prueba, knn_predictions)
sensitivity(y_prueba, knn_predictions)
specifity(y_prueba, knn_predictions)
f1(y_prueba, knn_predictions)

# ----- MAQUINAS VECTOR SOPORTE -----
print("----- MAQUINAS VECTOR SOPORTE -----")

svm_model = SVC(kernel='linear')
svm_model.fit(x_entrenamiento, y_entrenamiento)

svm_predictions = svm_model.predict(x_prueba)

accuracy(y_prueba, svm_predictions)
precision(y_prueba, svm_predictions)
sensitivity(y_prueba, svm_predictions)
specifity(y_prueba, svm_predictions)
f1(y_prueba, svm_predictions)

# ----- NAIVE BAYES -----
print("----- NAIVE BAYES -----")

nb_model = GaussianNB()
nb_model.fit(x_entrenamiento, y_entrenamiento)

# Realizar predicciones en el conjunto de prueba
nb_predictions = nb_model.predict(x_prueba)

# Calcular la precisión del modelo
accuracy(y_prueba, nb_predictions)
precision(y_prueba, nb_predictions)
sensitivity(y_prueba, nb_predictions)
specifity(y_prueba, nb_predictions)
f1(y_prueba, nb_predictions)