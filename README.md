# Redes neuronales con TensorFlow

Este repositorio contiene un ejemplo de red neuronal básica, utilizando el framework Keras de TensorFlow.

## Caso modelado

El archivo `red neuronal.py` contiene una función `MRU()` que genera valores de posición para cada valor de tiempo (en unidades arbitrarias), representando lo que sería una ecuación de movimiento rectilíneo uniforme: 

![equation](http://www.sciweavers.org/tex2img.php?eq=%20x_%7Bf%7D%20%3D%20x_%7Bi%7D%20%2Bvt&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

A partir de los valores de tiempo generados con `range()` y la salida de la función, se utiliza el framework Keras para generar un modelo el cual busca el mejor ajuste a los datos suministrados. Además se utiliza `pyplot` para graficar la función de pérdida del modelo.