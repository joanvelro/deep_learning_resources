from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

print(tf.__version__)

warnings.simplefilter(action='ignore', category=FutureWarning)
dataset_path = keras.utils.get_file("auto-mpg.data",
                                    "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")


column_names = ['MPG',
                'Cylinders',
                'Displacement',
                'Horsepower',
                'Weight',
                'Acceleration',
                'Model Year',
                'Origin']
raw_dataset = pd.read_csv(dataset_path,
                          names=column_names,
                          na_values="?",
                          comment='\t',
                          sep=" ",
                          skipinitialspace=True)

dataset = raw_dataset.copy()

dataset.isna().sum()
dataset = dataset.dropna()
"""
    La columna de "Origin" realmente es categorica, no numerica. Entonces conviertala a un "one-hot":
"""

y = dataset['MPG']
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1) * 1.0
dataset['Europe'] = (origin == 2) * 1.0
dataset['Japan'] = (origin == 3) * 1.0

dataset[['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year']].plot()
dataset[['MPG']].plot()

"""
    Dividamos la data en entrenamiento y prueba
    Ahora divida el set de datos en un set de entrenamiento y otro de pruebas.
    Usaremos el set de pruebas en la evaluacion final de nuestro modelo.

"""
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

"""
    Inspeccione la data
    Revise rapidamente la distribucion conjunta de un par de columnas de el set de entrenamiento.
"""
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")

"""
    Tambien revise las estadisticas generales:
"""
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
print(train_stats)

"""  
    Separe las caracteristicas de las etiquetas.
    Separe el valor objetivo, o la "etiqueta" de las caracteristicas. Esta etiqueta es el valor que entrenara el modelo para predecir.
"""
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

"""
    Normalice la data
    Revise otra vez el bloque de train_stats que se presento antes y note la diferencia de rangos de cada 
    caracteristica.
    Es una buena práctica normalizar funciones que utilizan diferentes escalas y rangos. Aunque el modelo * podría * 
    converger sin normalización de características, dificulta el entrenamiento y hace que el modelo resultante dependa
    de la elección de las unidades utilizadas en la entrada.
    Nota: Aunque generamos intencionalmente estas estadísticas solo del conjunto de datos de entrenamiento, estas 
    estadísticas también se utilizarán para normalizar el conjunto de datos de prueba. Necesitamos hacer eso para 
    proyectar el conjunto de datos de prueba en la misma distribución en la que el modelo ha sido entrenado.
    Estos datos normalizados es lo que usaremos para entrenar el modelo.
    Precaución: las estadísticas utilizadas para normalizar las entradas aquí (media y desviación estándar) deben 
    aplicarse a cualquier otro dato que se alimente al modelo, junto con la codificación de un punto que hicimos 
    anteriormente. Eso incluye el conjunto de pruebas, así como los datos en vivo cuando el modelo se usa en producción.
"""


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

"""
    El modelo
    Construye el modelo
    Construyamos nuestro modelo. Aquí, utilizaremos un modelo secuencial con dos capas ocultas densamente conectadas
    y una capa de salida que devuelve un único valor continuo. Los pasos de construcción del modelo se envuelven en
    una función, build_model, ya que crearemos un segundo modelo, más adelante.
"""


def build_model():
    model = keras.Sequential([
        layers.Dense(100, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(100, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


model = build_model()

"""
    Inspeccione el modelo
    Use el método .summary para imprimir una descripción simple del modelo
"""
model.summary()

"""
    Ahora pruebe el modelo. Tome un lote de ejemplos 10 de los datos de entrenamiento y llame amodel.predict en él.
"""
if 0 == 1:
    example_batch = normed_train_data[:9]
    example_result = model.predict(example_batch)
    example_result = example_result.ravel()
    print(example_result)

"""
    Display training progress by printing a single dot for each completed epoch
"""


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')


EPOCHS = 1000

"""
    Fit the model
"""
if 1 == 0:
    history = model.fit(
        normed_train_data,
        train_labels,
        epochs=EPOCHS,
        validation_split=0.2,
        verbose=0,
        callbacks=[PrintDot()])


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    # print('\n')
    # print(hist.columns)
    # print(hist.tail())

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()


"""
    Fit the model with early stop
"""
# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
model = build_model()
history = model.fit(normed_train_data,
                    train_labels,
                    epochs=EPOCHS,
                    validation_split=0.2,
                    verbose=0,
                    callbacks=[early_stop, PrintDot()])

plot_history(history)

""" 
    Evaluate    
"""
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

"""
    Make test predictions
"""

test_predictions = model.predict(normed_test_data).flatten()

"""
    Plot predictions against real values
"""
plt.figure()
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
plt.plot([-100, 100], [-100, 100])

"""
    Plot error distribution
"""
plt.figure()
error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel("Prediction Error [MPG]")
plt.ylabel("Count")

R2 = r2_score(test_predictions, test_labels)
MSE = mean_squared_error(test_predictions, test_labels)
print(R2)
print(MSE)


"""
    Make  predictions
"""
dataset.pop('MPG')
X = norm(dataset)
y_hat = model.predict(X).flatten()

plt.figure()
plt.plot(y.values, label='real values', color='red')
plt.plot(y_hat, label='prediction', color='blue')
plt.legend()

