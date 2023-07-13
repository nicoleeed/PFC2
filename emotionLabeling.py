import json
import os
import csv
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import cv2

from tensorflow.keras.models import load_model


model = load_model("ageLabelingCNN.h5")


# Crear una estructura de datos en formato COCO
coco_data = {
    "images": [],
    "age": [],
    "emotion": []
}

# Agregar información de las imágenes y anotaciones
images = []
age = []
emotion = []
image_id = 1

# Ruta de la carpeta que contiene las imágenes
images_folder = "/Users/nicole/Documents/2023-1/PFC2/Data Set/emotions/happy"

folder_name = os.path.basename(images_folder)

# Recorrer las imágenes en la carpeta
for filename in os.listdir(images_folder):
    if filename.endswith(".jpg"):
        # Leer la imagen
        image_path = os.path.join(images_folder, filename)
        image = Image.open(image_path)
        
        # Agregar información de la imagen a COCO
        image_info = {"id": image_id, "file_name": filename, "path": os.path.join(folder_name, filename)}
        coco_data["images"].append(image_info)

        # Leer la imagen
        imagen_prueba = cv2.imread(image_path)
        imagen_prueba = cv2.cvtColor(imagen_prueba, cv2.COLOR_BGR2GRAY)
        imagen_prueba = cv2.resize(imagen_prueba, (144, 144))
        imagen_prueba = np.expand_dims(imagen_prueba, axis=-1)
        imagen_prueba = imagen_prueba / 255.0
        imagen_prueba = np.expand_dims(imagen_prueba, axis=0)

        # Realizar la predicción
        prediccion = model.predict(imagen_prueba)

        # Obtener el valor de la predicción (probabilidad)
        probabilidad_kid = prediccion[0][0]
        probabilidad_no_kid = 1 - prediccion[0][0]

        if probabilidad_no_kid > probabilidad_kid :
            probabilidad = "no kid"
        elif probabilidad_kid > probabilidad_no_kid :
            probabilidad = "kid"

        # Agregar Edad a COCO
        coco_data["age"].append(probabilidad)

        # Agregar Edad a COCO
        coco_data["emotion"].append("happy")
        
        # Incrementar los IDs
        image_id += 1


# Ruta de la carpeta que contiene las imágenes
images_folder = "/Users/nicole/Documents/2023-1/PFC2/Data Set/emotions/sad"

folder_name = os.path.basename(images_folder)

# Recorrer las imágenes en la carpeta
for filename in os.listdir(images_folder):
    if filename.endswith(".jpg"):
        # Leer la imagen
        image_path = os.path.join(images_folder, filename)
        image = Image.open(image_path)
        
        # Agregar información de la imagen a COCO
        image_info = {"id": image_id, "file_name": filename, "path": os.path.join(folder_name, filename)}
        coco_data["images"].append(image_info)

        # Leer la imagen
        imagen_prueba = cv2.imread(image_path)
        imagen_prueba = cv2.cvtColor(imagen_prueba, cv2.COLOR_BGR2GRAY)
        imagen_prueba = cv2.resize(imagen_prueba, (144, 144))
        imagen_prueba = np.expand_dims(imagen_prueba, axis=-1)
        imagen_prueba = imagen_prueba / 255.0
        imagen_prueba = np.expand_dims(imagen_prueba, axis=0)

        # Realizar la predicción
        prediccion = model.predict(imagen_prueba)

        # Obtener el valor de la predicción (probabilidad)
        probabilidad_kid = prediccion[0][0]
        probabilidad_no_kid = 1 - prediccion[0][0]

        if probabilidad_no_kid > probabilidad_kid :
            probabilidad = "no kid"
        elif probabilidad_kid > probabilidad_no_kid :
            probabilidad = "kid"

        # Agregar Edad a COCO
        coco_data["age"].append(probabilidad)

        # Agregar Edad a COCO
        coco_data["emotion"].append("sad")

        # Incrementar los IDs
        image_id += 1


# Ruta de la carpeta que contiene las imágenes
images_folder = "/Users/nicole/Documents/2023-1/PFC2/Data Set/emotions/neutral"

folder_name = os.path.basename(images_folder)

# Recorrer las imágenes en la carpeta
for filename in os.listdir(images_folder):
    if filename.endswith(".jpg"):
        # Leer la imagen
        image_path = os.path.join(images_folder, filename)
        image = Image.open(image_path)
        
        # Agregar información de la imagen a COCO
        image_info = {"id": image_id, "file_name": filename, "path": os.path.join(folder_name, filename)}
        coco_data["images"].append(image_info)

        # Leer la imagen
        imagen_prueba = cv2.imread(image_path)
        imagen_prueba = cv2.cvtColor(imagen_prueba, cv2.COLOR_BGR2GRAY)
        imagen_prueba = cv2.resize(imagen_prueba, (144, 144))
        imagen_prueba = np.expand_dims(imagen_prueba, axis=-1)
        imagen_prueba = imagen_prueba / 255.0
        imagen_prueba = np.expand_dims(imagen_prueba, axis=0)

        # Realizar la predicción
        prediccion = model.predict(imagen_prueba)

        # Obtener el valor de la predicción (probabilidad)
        probabilidad_kid = prediccion[0][0]
        probabilidad_no_kid = 1 - prediccion[0][0]

        if probabilidad_no_kid > probabilidad_kid :
            probabilidad = "no kid"
        elif probabilidad_kid > probabilidad_no_kid :
            probabilidad = "kid"

        # Agregar Edad a COCO
        coco_data["age"].append(probabilidad)

        # Agregar Edad a COCO
        coco_data["emotion"].append("neutral")

        # Incrementar los IDs
        image_id += 1

# Ruta de la carpeta que contiene las imágenes
images_folder = "/Users/nicole/Documents/2023-1/PFC2/Data Set/emotions/angry"

folder_name = os.path.basename(images_folder)

# Recorrer las imágenes en la carpeta
for filename in os.listdir(images_folder):
    if filename.endswith(".jpg"):
        # Leer la imagen
        image_path = os.path.join(images_folder, filename)
        image = Image.open(image_path)
        
        # Agregar información de la imagen a COCO
        image_info = {"id": image_id, "file_name": filename, "path": os.path.join(folder_name, filename)}
        coco_data["images"].append(image_info)

        # Leer la imagen
        imagen_prueba = cv2.imread(image_path)
        imagen_prueba = cv2.cvtColor(imagen_prueba, cv2.COLOR_BGR2GRAY)
        imagen_prueba = cv2.resize(imagen_prueba, (144, 144))
        imagen_prueba = np.expand_dims(imagen_prueba, axis=-1)
        imagen_prueba = imagen_prueba / 255.0
        imagen_prueba = np.expand_dims(imagen_prueba, axis=0)

        # Realizar la predicción
        prediccion = model.predict(imagen_prueba)

        # Obtener el valor de la predicción (probabilidad)
        probabilidad_kid = prediccion[0][0]
        probabilidad_no_kid = 1 - prediccion[0][0]

        if probabilidad_no_kid > probabilidad_kid :
            probabilidad = "no kid"
        elif probabilidad_kid > probabilidad_no_kid :
            probabilidad = "kid"

        # Agregar Edad a COCO
        coco_data["age"].append(probabilidad)

        # Agregar Edad a COCO
        coco_data["emotion"].append("angry")

        # Incrementar los IDs
        image_id += 1


# Ruta de la carpeta que contiene las imágenes
images_folder = "/Users/nicole/Documents/2023-1/PFC2/Data Set/emotions/disgusted"

folder_name = os.path.basename(images_folder)

# Recorrer las imágenes en la carpeta
for filename in os.listdir(images_folder):
    if filename.endswith(".jpg"):
        # Leer la imagen
        image_path = os.path.join(images_folder, filename)
        image = Image.open(image_path)
        
        # Agregar información de la imagen a COCO
        image_info = {"id": image_id, "file_name": filename, "path": os.path.join(folder_name, filename)}
        coco_data["images"].append(image_info)

       # Leer la imagen
        imagen_prueba = cv2.imread(image_path)
        imagen_prueba = cv2.cvtColor(imagen_prueba, cv2.COLOR_BGR2GRAY)
        imagen_prueba = cv2.resize(imagen_prueba, (144, 144))
        imagen_prueba = np.expand_dims(imagen_prueba, axis=-1)
        imagen_prueba = imagen_prueba / 255.0
        imagen_prueba = np.expand_dims(imagen_prueba, axis=0)

        # Realizar la predicción
        prediccion = model.predict(imagen_prueba)

        # Obtener el valor de la predicción (probabilidad)
        probabilidad_kid = prediccion[0][0]
        probabilidad_no_kid = 1 - prediccion[0][0]

        if probabilidad_no_kid > probabilidad_kid :
            probabilidad = "no kid"
        elif probabilidad_kid > probabilidad_no_kid :
            probabilidad = "kid"

        # Agregar Edad a COCO
        coco_data["age"].append(probabilidad)

        # Agregar Edad a COCO
        coco_data["emotion"].append("disgusted")

        # Incrementar los IDs
        image_id += 1

# Ruta de la carpeta que contiene las imágenes
images_folder = "/Users/nicole/Documents/2023-1/PFC2/Data Set/emotions/fearful"

folder_name = os.path.basename(images_folder)

# Recorrer las imágenes en la carpeta
for filename in os.listdir(images_folder):
    if filename.endswith(".jpg"):
        # Leer la imagen
        image_path = os.path.join(images_folder, filename)
        image = Image.open(image_path)
        
        # Agregar información de la imagen a COCO
        image_info = {"id": image_id, "file_name": filename, "path": os.path.join(folder_name, filename)}
        coco_data["images"].append(image_info)

        # Leer la imagen
        imagen_prueba = cv2.imread(image_path)
        imagen_prueba = cv2.cvtColor(imagen_prueba, cv2.COLOR_BGR2GRAY)
        imagen_prueba = cv2.resize(imagen_prueba, (144, 144))
        imagen_prueba = np.expand_dims(imagen_prueba, axis=-1)
        imagen_prueba = imagen_prueba / 255.0
        imagen_prueba = np.expand_dims(imagen_prueba, axis=0)

        # Realizar la predicción
        prediccion = model.predict(imagen_prueba)

        # Obtener el valor de la predicción (probabilidad)
        probabilidad_kid = prediccion[0][0]
        probabilidad_no_kid = 1 - prediccion[0][0]

        if probabilidad_no_kid > probabilidad_kid :
            probabilidad = "no kid"
        elif probabilidad_kid > probabilidad_no_kid :
            probabilidad = "kid"

        # Agregar Edad a COCO
        coco_data["age"].append(probabilidad)

        # Agregar Edad a COCO
        coco_data["emotion"].append("fearful")

        # Incrementar los IDs
        image_id += 1


# Ruta de la carpeta que contiene las imágenes
images_folder = "/Users/nicole/Documents/2023-1/PFC2/Data Set/emotions/surprised"

folder_name = os.path.basename(images_folder)

# Recorrer las imágenes en la carpeta
for filename in os.listdir(images_folder):
    if filename.endswith(".jpg"):
        # Leer la imagen
        image_path = os.path.join(images_folder, filename)
        image = Image.open(image_path)
        
        # Agregar información de la imagen a COCO
        image_info = {"id": image_id, "file_name": filename, "path": os.path.join(folder_name, filename)}
        coco_data["images"].append(image_info)

        # Leer la imagen
        imagen_prueba = cv2.imread(image_path)
        imagen_prueba = cv2.cvtColor(imagen_prueba, cv2.COLOR_BGR2GRAY)
        imagen_prueba = cv2.resize(imagen_prueba, (144, 144))
        imagen_prueba = np.expand_dims(imagen_prueba, axis=-1)
        imagen_prueba = imagen_prueba / 255.0
        imagen_prueba = np.expand_dims(imagen_prueba, axis=0)

        # Realizar la predicción
        prediccion = model.predict(imagen_prueba)

        # Obtener el valor de la predicción (probabilidad)
        probabilidad_kid = prediccion[0][0]
        probabilidad_no_kid = 1 - prediccion[0][0]

        if probabilidad_no_kid > probabilidad_kid :
            probabilidad = "no kid"
        elif probabilidad_kid > probabilidad_no_kid :
            probabilidad = "kid"

        # Agregar Edad a COCO
        coco_data["age"].append(probabilidad)

        # Agregar Edad a COCO
        coco_data["emotion"].append("surprised")

        # Incrementar los IDs
        image_id += 1

# Guardar la estructura de datos en un archivo JSON
output_path = "/Users/nicole/Documents/2023-1/PFC2/Data Set/emotions/emotions_data.json"

with open(output_path, "w") as f:
    json.dump(coco_data, f)

# Guardar los datos en un archivo CSV
csv_filename = "/Users/nicole/Documents/2023-1/PFC2/Data Set/emotions/emotions_data.csv"
# Crear el archivo CSV
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["image_id", "file_name", "path", "age", "emotion"])

    # Recorrer las imágenes en coco_data
    for i, image_info in enumerate(coco_data["images"]):
        image_id = image_info["id"]
        file_name = image_info["file_name"]
        path = image_info["path"]
        age = coco_data["age"][i]
        emotion = coco_data["emotion"][i]

        # Escribir la información en el archivo CSV
        writer.writerow([image_id, file_name, path, age, emotion])

# Realizar el conteo de elementos para cada etiqueta
age_count = {}
for age_label in coco_data["emotion"]:
    age_count[age_label] = age_count.get(age_label, 0) + 1

# Imprimir la cantidad de elementos para cada etiqueta
print("Cantidad de elementos por etiqueta de emocion:")
for age_label, count in age_count.items():
    print(f"{age_label}: {count}")