import numpy as np
import pandas as pd
import warnings

"""este programa crea un modelo de red neuronal simple para predecir si un usuario realizará una compra, que tan probable es que compre un producto o servicio,
en función de la duración de su visita y el número de páginas visitadas en un sitio web. 
Utiliza TensorFlow y Keras para construir y entrenar el modelo."""



warnings.filterwarnings('ignore')  # Suprime advertencias no críticas

#**Generación de datos sintéticos**
np.random.seed(0)  # Fija semilla aleatoria para reproducibilidad
data_size = 200  # Número de muestras

# Crear características: 200 muestras, 2 características cada una
features = np.random.rand(data_size, 2)  # Valores uniformes en [0, 1)  # np.random.rand(m, n) --> Genera matriz m×n con valores aleatorios uniformes en [0, 1)
"""
features = [
    [0.5488135, 0.71518937],  # Ejemplo: fila 1
    [0.60276338, 0.54488318],  # Ejemplo: fila 2
    ... 200 filas
]
"""




# Crear etiquetas: 1 si suma de características > 1, 0 en otro caso
labels = (features[:, 0] + features[:, 1] > 1).astype(int) #features[:, 0] --> Selecciona todas las filas de la columna 0 (duración de visita)
"""
Ejemplo:
   [0.55 + 0.72 = 1.27 > 1 → 1]
   [0.60 + 0.54 = 1.14 > 1 → 1]
   [0.42 + 0.37 = 0.79 < 1 → 0]
"""

# Convertir a DataFrame
df = pd.DataFrame(features, columns=['VisitDuration', 'PagesVisited'])
df['Purchase'] = labels  # Añadir columna de compras

##El DataFrame resultante tendrá 3 columnas y 200 filas



#**Preprocesamiento de datos**

from sklearn.model_selection import train_test_split

# Dividir datos en características (X) y objetivo (y)
X = df[['VisitDuration', 'PagesVisited']]  # 200×2
y = df['Purchase']  # Vector de 200 etiquetas

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% para prueba (40 muestras)
    random_state=42     # Semilla para reproducibilidad
)



#**Construccion y entrenamiento de la red neuronal**

import tensorflow as tf # framework para deep learning
from tensorflow.keras.models import Sequential #Imports the Sequential model, which allows stacking layers linearly.
from tensorflow.keras.layers import Dense # Imports the Dense layer, which is a fully connected layer in the neural network.

# Definir la arquitectura del modelo
model = Sequential([  # Modelo secuencial (capas en serie, son creadas una tras otra, en orden)
    # Capa oculta: 10 neuronas. Uses ReLU (Rectified Linear Unit) activation function--> For positive inputs, it's an identity function (output equals input).
    Dense(10, activation='relu', input_shape=(2,)),     # ... For negative inputs, it outputs zero.
    # ... input_shape expects input vectors of 2 features per sample.

    # Capa de salida: 1 neurona (binaria), activación sigmoide. En el contexto de redes neuronales, la función sigmoidea o función logística ...
    Dense(1, activation='sigmoid')      # ... es una función de activación que transforma los valores de entrada en un rango entre 0 y 1
])

# Compilar el modelo (configurar aprendizaje)
model.compile(
    optimizer='adam',  # Algoritmo de optimización adaptativo segun cada uno de sus parámetros
    loss='binary_crossentropy',  # Función de pérdida para binaria, mide la diferencia entre predicciones y las etiquetas reales
    metrics=['accuracy']  # Seguimiento de precisión durante entrenamiento y evaluación
)



#**Entrenamiento del modelo**
import matplotlib.pyplot as plt

# Train the model with validation split and save the history
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=10,
    validation_split=0.2
)

# Plot training and validation accuracy and loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()



#**Evaluación del modelo**
# Evaluar en conjunto de prueba
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")

# (Opcional: Imprimir pérdida también)
print(f"Test Loss: {test_loss}")


""" Interpreting the results!!!:

accuracy: Proportion of correct predictions on the training data.
val_accuracy: Proportion of correct predictions on the validation set.

loss: How far the model’s predictions are from the true values (lower is better).
val_loss: Prediction error on the validation set

Train: Measures performance on the data used to fit the model.
Validation: Measures performance on unseen data during training, helping detect overfitting.

If both training and validation accuracy increase and loss decreases, the model is learning well.
If training accuracy is much higher than validation accuracy, the model may be overfitting (memorizing training data, not generalizing).
If both are low, the model is underfitting (not learning enough).
"""



"""

Key concepts:  

Neural Network: A computational model inspired by the human brain, consisting of layers of interconnected nodes (neurons).

Sequential model: Simple stack of layers, no branching.

Dense layer: Each neuron receives input from all outputs of the previous layer.

Activation functions: Add non-linearity (ReLU for hidden, sigmoid for output).

Optimizer: Algorithm to update weights (Adam is adaptive and popular).

Loss function: Guides learning by quantifying prediction error.

Epoch: One full pass through the training data.

Batch size: Number of samples processed before updating the model.

ReLu: Activation function that outputs the input directly if positive, otherwise zero. Helps with non-linearity and avoids vanishing gradient problem.

Sigmoid: Activation function that transforms inputs to a range between 0 and 1, useful for binary classification.

"""