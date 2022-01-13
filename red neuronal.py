#   1   -   Importaciones necesarias
import tensorflow as tf, matplotlib.pyplot as plt, numpy as np
from tensorflow.keras.layers import Dense
from keras.models import Sequential
from tensorflow.python.keras.backend import dtype

#   2   -   Función que define movimiento rectilíneo uniforme
def MRU(posicion_inicial, velocidad_inicial, tiempo):
    posicion = []
    for t in tiempo:
        posicion_final = posicion_inicial + velocidad_inicial * t
        posicion.append(posicion_final)
    return np.asarray(posicion)

#   3   -   Listas que almacenan valores de tiempo, y posición en función del tiempo
valores_tiempo = np.asarray(range(100))
valores_posicion = MRU(0, 50, valores_tiempo)

#   4   -   Creación del modelo mediante el framework Keras
#           Pueden ajustarse los valores de tasa de aprendizaje y número de épocas para obtener un modelo más preciso
#           Si el valor de tasa de aprendizaje es más chico, el modelo aprende más lento y se necesitan más épocas
capa = tf.keras.layers.Dense(units = 1, input_shape = [1])
modelo = tf.keras.Sequential([capa])
modelo.compile(optimizer = tf.keras.optimizers.Adam(0.05), loss = 'mean_squared_error')  # optimizador y función de pérdida
historial = modelo.fit(valores_tiempo, valores_posicion, epochs = 5000, verbose = False) # entrenamiento del modelo

#   5   -   Gráfico de la función de pérdida para ajustar los ciclos de entrenamiento.
plt.xlabel("Época")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])
plt.show()

#   6   -   Verificación del modelo
posicion_a_tiempo_10 = modelo.predict([10])
print("La posición a tiempo 10 según el modelo es " + str(posicion_a_tiempo_10))
#   La posición obtenida es 500. El modelo posee un alto grado de ajuste