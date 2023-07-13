import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import csv
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Lambda 
from keras.layers import Concatenate
from keras.backend import int_shape, flatten
from keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Input, Reshape, Activation
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import LabelEncoder
from scipy.ndimage import gaussian_filter
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization
from sklearn.metrics import accuracy_score, f1_score
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/nicole/Documents/2023-1/PFC2/propuesta/shape_predictor_68_face_landmarks.dat")

def get_facial_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray)
    result=image
    for face in faces:
        landmarks = predictor(gray, face)
        landmarks_dict = {}

        landmarks_dict['eye_left'] = (landmarks.part(36).x, landmarks.part(36).y)
        landmarks_dict['eye_right'] = (landmarks.part(45).x, landmarks.part(45).y)
        landmarks_dict['eyebrow_left'] = (landmarks.part(17).x, landmarks.part(19).y)
        landmarks_dict['eyebrow_right'] = (landmarks.part(26).x, landmarks.part(24).y)
        landmarks_dict['nose_base'] = (landmarks.part(33).x, landmarks.part(33).y)
        landmarks_dict['nose_tip'] = (landmarks.part(30).x, landmarks.part(30).y)
        landmarks_dict['nose_left'] = (landmarks.part(31).x, landmarks.part(31).y)
        landmarks_dict['nose_right'] = (landmarks.part(35).x, landmarks.part(35).y)
        landmarks_dict['lip_upper_center'] = (landmarks.part(51).x, landmarks.part(51).y)
        landmarks_dict['lip_lower_center'] = (landmarks.part(57).x, landmarks.part(57).y)
        landmarks_dict['lip_left_corner'] = (landmarks.part(48).x, landmarks.part(48).y)
        landmarks_dict['lip_right_corner'] = (landmarks.part(54).x, landmarks.part(54).y)
        landmarks_dict['cheek_left'] = (landmarks.part(1).x, landmarks.part(1).y)
        landmarks_dict['cheek_right'] = (landmarks.part(15).x, landmarks.part(15).y)


        # Dibujar los landmarks y sus alrededores en la imagen original
        radius = 20  # Radio para resaltar los landmarks
        mask = np.zeros_like(image)
        
        for landmark in landmarks_dict.values():
            cv2.circle(mask, landmark, radius, (255, 255, 255), -1)
 
        # Aplicar la máscara a la imagen original
        result = cv2.bitwise_and(image, mask)

    return result


def detect_skin_tone(image):
    # Convertir la imagen a espacio de color HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Definir los valores de umbral para detección de tono de piel en HSV
    lower_threshold = np.array([0, 20, 70], dtype=np.uint8)
    upper_threshold = np.array([25, 255, 255], dtype=np.uint8)

    # Aplicar la detección de tono de piel utilizando los valores de umbral
    skin_tone_mask = cv2.inRange(hsv_image, lower_threshold, upper_threshold)

    # Aplicar operaciones de morfología para mejorar la detección
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_tone_mask = cv2.morphologyEx(skin_tone_mask, cv2.MORPH_OPEN, kernel)
    skin_tone_mask = cv2.morphologyEx(skin_tone_mask, cv2.MORPH_CLOSE, kernel)

    # Obtener la imagen con el tono de piel detectado
    skin_tone_image = cv2.bitwise_and(image, image, mask=skin_tone_mask)

    return skin_tone_image

def detect_circles(image):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar suavizado a la imagen para reducir ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detección de círculos utilizando el algoritmo de Hough
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                               param1=50, param2=30, minRadius=5, maxRadius=50)

    # Si se encontraron círculos, eliminar el fondo
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        mask = np.zeros_like(gray)
        for (x, y, r) in circles:
            cv2.circle(mask, (x, y), r, (255), -1)

        masked_image = cv2.bitwise_and(image, image, mask=mask)
        return masked_image

    return image

def remove_background(image):
    # Obtener el número de canales de la imagen
    num_channels = len(image.shape)

    if num_channels == 2:
        print( "Blanco y negro")
        image=detect_skin_tone(image)
        image=detect_circles(image)
    elif num_channels == 3:
        print( "A color")
    else:
        print( "Formato de imagen no válido")

    return image
  

# Ruta del archivo CSV
coco_csv_path = "/Users/nicole/Documents/2023-1/PFC2/Data Set/emotions/train_emotions_data_70.csv"
max_images_per_label = 1000

# Obtener las imágenes y las etiquetas del archivo CSV
images = []
emotions = []
ages = []

# Leer el archivo CSV
with open(coco_csv_path, 'r') as file:
    reader = csv.reader(file)
    header = next(reader)  # Leer la primera fila del encabezado

    for row in reader:
        image_id = row[0]
        file_name = row[1]
        path = row[2]
        age = row[3]
        emotion = row[4]

        images.append({
            "id": image_id,
            "file_name": file_name,
            "path": path
        })
        emotions.append(emotion)
        ages.append(age)

# Contador de imágenes por etiqueta
label_counts = {}
age_counts = {}

# Crear las listas para almacenar los datos de entrenamiento y prueba
X = []
y = []

# Ruta de la carpeta que contiene las imágenes
images_folder = "/Users/nicole/Documents/2023-1/PFC2/Data Set/emotions"

# Leer las imágenes y las etiquetas
for image_info, label, age in zip(images, emotions, ages):
    if label not in label_counts:
        label_counts[label] = 0
        age_counts[label] = {"kid": 0, "no kid": 0}

    # Verificar el límite de imágenes por etiqueta
    if label_counts[label] >= max_images_per_label:
        continue

    # Clasificar las imágenes según la etiqueta de edad
    if age == "kid":
        if age_counts[label]["kid"] < 1 * max_images_per_label:
            age_counts[label]["kid"] += 1
        else:
            continue
    else:
        if age_counts[label]["no kid"] < 0 * max_images_per_label:
            age_counts[label]["no kid"] += 1
        else:
            continue

    # Ruta completa de la imagen
    image_file = os.path.join(images_folder, str(image_info["path"]))

    # Leer la imagen
    image = cv2.imread(image_file)
    # Redimensionar a 48x48 píxeles y convertir a escala de grises
    image = cv2.resize(image, (144, 144))
    #image = remove_background(image)
    image = get_facial_landmarks(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Incrementar el contador de imágenes por etiqueta
    label_counts[label] += 1

    X.append(image)
    y.append(label)

# Realizar el conteo de elementos para cada etiqueta
for emotion, count in label_counts.items():
    print(f"{emotion}: {count} elementos (niños: {age_counts[emotion]['kid']}, no niños: {age_counts[emotion]['no kid']})")

# Convertir las listas a matrices numpy
X = np.array(X)
y = np.array(y)

# Expandir las dimensiones de X para agregar el canal de color
X = np.expand_dims(X, axis=-1)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, y_train = X, y
X_train = X_train.astype(np.uint8)
X_train = X_train * 255

# Ruta del archivo CSV de prueba
coco_csv_path_test = "/Users/nicole/Documents/2023-1/PFC2/Data Set/emotions/test_emotions_data_30.csv"

# Obtener las imágenes y las etiquetas del archivo CSV de prueba
images_ = []
emotions_ = []
ages_ = []

# Leer el archivo CSV de prueba
with open(coco_csv_path_test, 'r') as file:
    reader = csv.reader(file)
    header = next(reader)  # Leer la primera fila del encabezado

    for row in reader:
        image_id = row[0]
        file_name = row[1]
        path = row[2]
        age = row[3]
        emotion = row[4]

        images_.append({
            "id": image_id,
            "file_name": file_name,
            "path": path
        })
        emotions_.append(emotion)
        ages_.append(age)

# Contador de imágenes por etiqueta
label_counts_ = {}
age_counts_ = {}

# Crear las listas para almacenar los datos de prueba
X_ = []
y_ = []
max_images_per_label=450
# Leer las imágenes y las etiquetas del archivo CSV de prueba
for image_info, label, age in zip(images_, emotions_, ages_):
    if label not in label_counts_:
        label_counts_[label] = 0
        age_counts_[label] = {"kid": 0, "no kid": 0}

    # Verificar el límite de imágenes por etiqueta
    if label_counts_[label] >= max_images_per_label:
        continue

    # Clasificar las imágenes según la etiqueta de edad
    if age == "kid":
        if age_counts_[label]["kid"] < 1 * max_images_per_label:
            age_counts_[label]["kid"] += 1
        else:
            continue
    else:
        if age_counts_[label]["no kid"] < 0.0 * max_images_per_label:
            age_counts_[label]["no kid"] += 1
        else:
            continue

    # Ruta completa de la imagen
    image_file = os.path.join(images_folder, str(image_info["path"]))

    # Leer la imagen
    image = cv2.imread(image_file)
    # Redimensionar a 48x48 píxeles y convertir a escala de grises
    image = cv2.resize(image, (144, 144))
    #image = remove_background(image)
    image = get_facial_landmarks(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Incrementar el contador de imágenes por etiqueta
    label_counts_[label] += 1

    X_.append(image)
    y_.append(label)

# Realizar el conteo de elementos para cada etiqueta
for emotion, count in label_counts_.items():
    print(f"{emotion}: {count} elementos (niños: {age_counts_[emotion]['kid']}, no niños: {age_counts_[emotion]['no kid']})")

X_ = np.array(X_)
y_ = np.array(y_)

# Expandir las dimensiones de X para agregar el canal de color
X_ = np.expand_dims(X_, axis=-1)

X_test, y_test = X_, y_
X_test = X_test.astype(np.uint8)
X_test = X_test * 255



num_classes = len(np.unique(y))

# Inicializar el codificador de etiquetas
label_encoder = LabelEncoder()

# Codificar las etiquetas de entrenamiento
y_train_encoded = label_encoder.fit_transform(y_train)

# Codificar las etiquetas de prueba (si es necesario)
y_test_encoded = label_encoder.fit_transform(y_test)

# Convertir las etiquetas codificadas a codificación one-hot
y_train = to_categorical(y_train_encoded, num_classes)
y_test = to_categorical(y_test_encoded, num_classes)

# Verificar las dimensiones de los conjuntos
print('Dimensiones de X_train:', X_train.shape)
print('Dimensiones de X_test:', X_test.shape)
print('Dimensiones de y_train:', y_train.shape)
print('Dimensiones de y_test:', y_test.shape)


# Matrices de filtro para la detección de bordes
top_edge_filter = np.array([[-1, -1, -1], [1, 1, 1], [0, 0, 0]]).reshape((3, 3, 1, 1))
bottom_edge_filter = np.array([[1, 1, 1], [0, 0, 0],[-1, -1, -1]]).reshape((3, 3, 1, 1))
left_edge_filter = np.array([[-1, 1, 0], [-1, 1, 0], [-1, 1, 0]]).reshape((3, 3, 1, 1))
right_edge_filter = np.array([[0, 1, -1], [0, 1, -1], [0, 1, -1]]).reshape((3, 3, 1, 1))



input_shape = (144, 144, 1)
inputs = Input(shape=input_shape)

conv_1=Conv2D(filters=1, kernel_size=(3, 3), padding="same", strides=(1, 1), kernel_regularizer=l2(0.001), kernel_initializer=tf.constant_initializer(top_edge_filter))(inputs)
conv_2=Conv2D(filters=1, kernel_size=(3, 3), padding="same", strides=(1, 1), kernel_regularizer=l2(0.001), kernel_initializer=tf.constant_initializer(bottom_edge_filter))(conv_1)
conv_3=Conv2D(filters=1, kernel_size=(3, 3), padding="same", strides=(1, 1), kernel_regularizer=l2(0.001), kernel_initializer=tf.constant_initializer(left_edge_filter))(conv_2)
conv_4=Conv2D(filters=1, kernel_size=(3, 3), padding="same", strides=(1, 1), kernel_regularizer=l2(0.001), kernel_initializer=tf.constant_initializer(right_edge_filter))(conv_3)

conv_5=Conv2D(filters=64, kernel_size=(3,3),padding="same",strides=(1,1),kernel_regularizer=l2(0.001))(conv_4)
actv_1=Activation('relu')(conv_5)
batchnorm_1 = BatchNormalization()(actv_1)
conv_6=Conv2D(filters=64, kernel_size=(3,3),padding="same",strides=(1,1),kernel_regularizer=l2(0.001))(batchnorm_1)
actv_2=Activation('relu')(conv_6)
batchnorm_2 = BatchNormalization()(actv_2)
maxp_1=MaxPool2D(pool_size=(2,2))(batchnorm_2)
drop_1=Dropout(0.1)(maxp_1)

conv_7=Conv2D(filters=128, kernel_size=(3,3),padding="same",strides=(1,1),kernel_regularizer=l2(0.001))(drop_1)
actv_3=Activation('relu')(conv_7)
batchnorm_3 = BatchNormalization()(actv_3)
conv_8=Conv2D(filters=128, kernel_size=(3,3),padding="same",strides=(1,1),kernel_regularizer=l2(0.001))(batchnorm_3)
actv_4=Activation('relu')(conv_8)
batchnorm_4 = BatchNormalization()(actv_4)
conv_9=Conv2D(filters=128, kernel_size=(3,3),padding="same",strides=(1,1),kernel_regularizer=l2(0.001))(batchnorm_4)
actv_5=Activation('relu')(conv_9)
batchnorm_5 = BatchNormalization()(actv_5)
maxp_2=MaxPool2D(pool_size=(2,2))(batchnorm_5)
drop_2=Dropout(0.1)(maxp_2)

conv_10=Conv2D(filters=256, kernel_size=(3,3),padding="same",strides=(1,1),kernel_regularizer=l2(0.001))(drop_2)
actv_6=Activation('relu')(conv_10)
batchnorm_6 = BatchNormalization()(actv_6)
conv_11=Conv2D(filters=256, kernel_size=(3,3),padding="same",strides=(1,1),kernel_regularizer=l2(0.001))(batchnorm_6)
actv_7=Activation('relu')(conv_11)
batchnorm_7 = BatchNormalization()(actv_7)
conv_12=Conv2D(filters=256, kernel_size=(3,3),padding="same",strides=(1,1),kernel_regularizer=l2(0.001))(batchnorm_7)
actv_8=Activation('relu')(conv_12)
batchnorm_8 = BatchNormalization()(actv_8)
conv_13=Conv2D(filters=256, kernel_size=(3,3),padding="same",strides=(1,1),kernel_regularizer=l2(0.001))(batchnorm_8)
actv_9=Activation('relu')(conv_13)
batchnorm_9 = BatchNormalization()(actv_9)
maxp_3=MaxPool2D(pool_size=(2,2))(batchnorm_9)
drop_3=Dropout(0.1)(maxp_3)
flat_1= Flatten()(drop_3)
outputs = Dense(7, activation='softmax')(flat_1)
model = Model(inputs=inputs, outputs=outputs)

tf.keras.utils.plot_model(model, show_shapes=True)
print(model.summary())
# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=64)

# Evaluar el modelo en el conjunto de prueba
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

# Calcular el accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calcular el F1 score
f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 Score:", f1)

# Imprimir el reporte de clasificación y la matriz de confusión
print(classification_report(y_test, y_pred))
confusion_mtx = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mtx, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Guardar el modelo entrenado
model.save('fercChilds100.h5')
