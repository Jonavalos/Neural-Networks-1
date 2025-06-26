import numpy as np
import pandas as pd
import warnings

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

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Definir la arquitectura del modelo
model = Sequential([  # Modelo secuencial (capas en serie)
    # Capa oculta: 10 neuronas, activación ReLU
    Dense(10, activation='relu', input_shape=(2,)),

    # Capa de salida: 1 neurona (binaria), activación sigmoide
    Dense(1, activation='sigmoid')
])

# Compilar el modelo (configurar aprendizaje)
model.compile(
    optimizer='adam',  # Algoritmo de optimización adaptativo
    loss='binary_crossentropy',  # Función de pérdida para binaria
    metrics=['accuracy']  # Seguimiento de precisión durante entrenamiento
)

# Entrenar el modelo
history = model.fit(
    X_train, y_train,  # Datos de entrenamiento
    epochs=10,  # Pasadas completas sobre los datos
    batch_size=10  # Tamaño de lote (10 muestras por actualización)
)



#**Evaluación del modelo**
# Evaluar en conjunto de prueba
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")

# (Opcional: Imprimir pérdida también)
print(f"Test Loss: {test_loss}")