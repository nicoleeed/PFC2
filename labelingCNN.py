import matplotlib.pyplot as plt
import json
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Lambda 
from keras.layers import Concatenate
from keras.backend import int_shape, flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Reshape, Activation
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import l2


# Ruta del archivo COCO en formato JSON
coco_json_path = '/Users/nicole/Documents/2023-1/PFC2/Data Set/age/age_data.json'

# Ruta de la carpeta que contiene las imágenes
images_folder = "/Users/nicole/Documents/2023-1/PFC2/Data Set/age/"

# Límite máximo por etiqueta
max_images_per_label = 22000


# Leer el archivo JSON
with open(coco_json_path, 'r') as f:
    coco_data = json.load(f)

# Lista para almacenar las imágenes y las etiquetas
imagenes = []
etiquetas = []
cont1=0
cont2=0
# Recorrer las imágenes en el archivo COCO
for image_info in coco_data['images']:
    imagen_path = image_info['path']

    # Obtener la etiqueta correspondiente a la imagen actual
    etiqueta = coco_data['age'][image_info['id'] - 1]

    if etiqueta == "kid" and cont1 >= max_images_per_label:
        continue
    elif etiqueta != "kid" and cont2 >= max_images_per_label:
        continue
    
    if cont1 >= max_images_per_label and cont2 >= max_images_per_label:
        break

    # Cargar la imagen
    imagen = cv2.imread(os.path.join(images_folder, imagen_path))
    
    imagen = cv2.resize(imagen, (144, 144))
    
    # Agregar la imagen y la etiqueta a las listas
    imagenes.append(imagen)
    etiquetas.append(etiqueta)

    if etiqueta == "kid" :
        cont1 += 1
    else:
        cont2 += 1


print(cont1)
print(cont2)
# Convertir las listas a arreglos numpy
imagenes = np.array(imagenes)
etiquetas = np.array(etiquetas)

# Imprimir la cantidad de elementos para cada etiqueta después de eliminar los excedentes
unique_labels, label_counts = np.unique(etiquetas, return_counts=True)
for label, count in zip(unique_labels, label_counts):
    print(f"Etiqueta: {label}, Cantidad de elementos: {count}")


# Dividir el dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(imagenes, etiquetas, test_size=0.3, random_state=42)

# Convertir imágenes a escala de grises
X_train = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in X_train])
X_test = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in X_test])


X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Normalizar los valores de los píxeles entre 0 y 1
X_train = X_train / 255.0
X_test = X_test / 255.0


label_mapping = {'kid': 1, 'no kid': 0}
y_train = np.array([label_mapping[label] for label in y_train])
y_test = np.array([label_mapping[label] for label in y_test])

print('Dimensiones de X_train:', X_train.shape)
print('Dimensiones de X_test:', X_test.shape)
print('Dimensiones de y_train:', y_train.shape)
print('Dimensiones de y_test:', y_test.shape)

input_shape=(144, 144, 1)

inputs = Input(shape=input_shape)
conv_1=Conv2D(filters=32, kernel_size=(3,3),padding="same",strides=(1,1),kernel_regularizer=l2(0.001))(inputs)
drop_1=Dropout(0.1)(conv_1)
actv_1=Activation('relu')(drop_1)
maxp_1=MaxPool2D(pool_size=(2,2))(actv_1)

conv_2=Conv2D(filters=64, kernel_size=(3,3),padding="same",strides=(1,1),kernel_regularizer=l2(0.001))(maxp_1)
drop_2=Dropout(0.1)(conv_2)
actv_2=Activation('relu')(drop_2)
maxp_2=MaxPool2D(pool_size=(2,2))(actv_2)

conv_3=Conv2D(filters=128, kernel_size=(3,3),padding="same",strides=(1,1),kernel_regularizer=l2(0.001))(maxp_2)
drop_3=Dropout(0.1)(conv_3)
actv_3=Activation('relu')(drop_3)
maxp_3=MaxPool2D(pool_size=(2,2))(actv_3)

conv_4=Conv2D(filters=256, kernel_size=(3,3),padding="same",strides=(1,1),kernel_regularizer=l2(0.001))(maxp_3)
drop_4=Dropout(0.1)(conv_4)
actv_4=Activation('relu')(drop_4)
maxp_4=MaxPool2D(pool_size=(2,2))(actv_4)

conv_5=Conv2D(filters=512, kernel_size=(3,3),padding="same",strides=(1,1),kernel_regularizer=l2(0.001))(maxp_4)
drop_5=Dropout(0.1)(conv_5)
actv_5=Activation('relu')(drop_5)
maxp_5=MaxPool2D(pool_size=(2,2))(actv_5)


flatten = Flatten()(maxp_5)
dense_1 = Dense(256, activation='relu')(flatten)
drop_1 = Dropout(0.2)(dense_1)

output_1 = Dense(1, activation='sigmoid', name='age')(drop_1)

model = Model(inputs=inputs, outputs=output_1)
tf.keras.utils.plot_model(model, show_shapes=True)


model.compile(loss=['binary_crossentropy','mae'], optimizer='adam',  metrics=['accuracy'])

# Entrenar el modelo 
history=model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))

# Evaluar el modelo en el conjunto de prueba
loss, accuracy = model.evaluate(X_test, y_test)

print("Precisión en el conjunto de prueba:", accuracy)

# Guardar el modelo entrenado
model.save("ageLabelingCNN.h5")

# plot results for age
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Loss Graph')
plt.legend()
plt.show()
