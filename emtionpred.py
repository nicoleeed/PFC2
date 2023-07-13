import matplotlib.pyplot as plt
import json
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import csv
import cv2
import os
from sklearn.metrics import f1_score
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score, f1_score


import dlib

import mplcyberpunk
plt.style.use("cyberpunk")

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

model = load_model("fercChilds.h5")
print("Train 70% kids, 30% no kids, Test 100% kids ")

# Ruta de la carpeta que contiene las imágenes
images_folder = "/Users/nicole/Documents/2023-1/PFC2/Data Set/emotions"


coco_csv_path_test = "/Users/nicole/Documents/2023-1/PFC2/Data Set/emotions/test1.csv"

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
max_images_per_label=500
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

num_classes = len(np.unique(y_))

# Inicializar el codificador de etiquetas
label_encoder = LabelEncoder()

# Codificar las etiquetas de prueba (si es necesario)
y_test_encoded = label_encoder.fit_transform(y_test)


y_test = to_categorical(y_test_encoded, num_classes)

# Verificar las dimensiones de los conjuntos
print('Dimensiones de X_test:', X_test.shape)
print('Dimensiones de y_test:', y_test.shape)

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

# Crear una lista con los nombres de las etiquetas originales
label_names = label_encoder.inverse_transform(range(num_classes))

# Imprimir el reporte de clasificación y la matriz de confusión
print(classification_report(y_test, y_pred))
confusion_mtx = confusion_matrix(y_test, y_pred)



plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='plasma') 
mplcyberpunk.add_glow_effects()
plt.xlabel('Predicted')
plt.ylabel('True')

x_ticks = np.arange(len(label_names))

# Reemplazar los valores en los ejes x e y por los nombres de las etiquetas reales
plt.xticks(x_ticks + 0.5, label_names, rotation=90)
plt.yticks(x_ticks + 0.5, label_names, rotation=0)
plt.show()

# Calcular el accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calcular el F1 score
f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 Score:", f1)

# Guardar las predicciones erróneas en un archivo CSV
errors = []
for i in range(len(y_test)):
    if y_test[i] != y_pred[i]:
        image_info = images_[i]
        image_id = image_info["id"]
        file_name = image_info["file_name"]
        path = image_info["path"]
        age = ages_[i]
        true_label = label_encoder.inverse_transform([y_test[i]])[0]
        predicted_label = label_encoder.inverse_transform([y_pred[i]])[0]
        errors.append([image_id, file_name, path, age, true_label, predicted_label])

# Ruta del archivo CSV de errores
errors_csv_path = "/Users/nicole/Documents/2023-1/PFC2/Data Set/emotions/errorschild.csv"

# Escribir los errores en el archivo CSV
with open(errors_csv_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["image_id", "file_name", "path", "age", "true_label", "predicted_label"])
    writer.writerows(errors)


