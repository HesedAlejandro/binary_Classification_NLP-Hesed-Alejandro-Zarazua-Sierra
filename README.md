# README

## Descripción

Este proyecto entrena y evalúa un modelo de red neuronal para la clasificación binaria de reseñas de películas utilizando el conjunto de datos IMDB. El flujo principal del programa está dividido en dos partes:

1. **Entrenamiento y evaluación del modelo**: Se entrena un modelo de red neuronal utilizando los datos de IMDB, luego se evalúa su rendimiento.
2. **Visualización de los resultados**: Se generan gráficos que muestran cómo la pérdida y la exactitud cambian durante las épocas de entrenamiento.

Este script se divide en varias funciones, y el archivo principal ejecuta el flujo del entrenamiento y evaluación llamando a una función externa (`train_and_evaluate_model`) desde un módulo separado.

## Requisitos

Antes de ejecutar este código, asegúrate de tener instaladas las siguientes librerías:

- **NumPy**: `pip install numpy`
- **Matplotlib**: `pip install matplotlib`
- **Keras**: `pip install keras`
- **TensorFlow**: `pip install tensorflow`

## Estructura del Proyecto

El código está estructurado de la siguiente manera:

```
src/
│
└───binary_classification_NLP.py  # Módulo que contiene la función train_and_evaluate_model
main.py  # Script principal para ejecutar el flujo del modelo
```

### 1. **Script principal (`main.py`)**

Este script ejecuta el flujo completo del proceso de entrenamiento y evaluación del modelo.

#### Flujo del Script:

1. Se importa la función `train_and_evaluate_model` desde el módulo `binary_classification_NLP`.
2. La función `main()` imprime un mensaje de inicio, ejecuta el proceso de entrenamiento y evaluación del modelo, y finalmente imprime un mensaje de completado.

```python
def main():
    """
    Función principal que ejecuta el entrenamiento y evaluación del modelo.
    """
    print("Iniciando el entrenamiento y evaluación del modelo...")
    train_and_evaluate_model()  # Llamar a la función para entrenar y evaluar el modelo
    print("Proceso completado.")
```

3. La función `main()` es ejecutada cuando el script es ejecutado directamente.

```python
if __name__ == "__main__":
    main()  # Ejecutar la función main() cuando el script se ejecuta directamente
```

### 2. **Función de Entrenamiento y Evaluación (`train_imdb_model`)**

La función `train_imdb_model()` entrena un modelo de red neuronal utilizando el conjunto de datos IMDB. Este modelo es capaz de clasificar las reseñas como positivas o negativas.

#### Flujo del Entrenamiento:

1. **Carga de datos**: Se cargan los datos de entrenamiento y prueba del conjunto IMDB y se realizan preprocesamientos como la vectorización de las secuencias de texto.
2. **Construcción del modelo**: Se construye un modelo de red neuronal con 2 capas ocultas y una capa de salida con activación sigmoidea para la clasificación binaria.
3. **Entrenamiento**: El modelo se entrena con un optimizador **RMSprop**, utilizando la función de pérdida **binary_crossentropy**.
4. **Evaluación**: El modelo se evalúa en el conjunto de prueba.
5. **Visualización**: Se generan gráficos que muestran la evolución de la pérdida y la exactitud durante el entrenamiento y validación.

```python
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```

#### Resultados:

El modelo muestra los gráficos de:

- **Pérdida de entrenamiento y validación**.
- **Exactitud de entrenamiento y validación**.

### 3. **Módulo `binary_classification_NLP.py`**

Este módulo contiene la función `train_and_evaluate_model`, que es responsable de ejecutar todo el flujo del modelo.

```python
def train_and_evaluate_model():
    model, x_test, y_test = train_imdb_model()
    # Se pueden agregar más funcionalidades, como guardar el modelo o realizar predicciones.
```

## Ejecución

Para ejecutar este proyecto, simplemente corre el script principal (`main.py`) desde la línea de comandos:

```bash
python main.py
```

Esto ejecutará el entrenamiento y evaluación del modelo, mostrando los resultados de evaluación y las gráficas correspondientes.

## Conclusión

Este proyecto proporciona una implementación simple de un clasificador binario utilizando redes neuronales para el análisis de sentimientos en reseñas de películas. Utiliza el conjunto de datos IMDB para entrenar el modelo, y presenta gráficos útiles para evaluar su rendimiento.