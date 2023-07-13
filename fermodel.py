from fer import FER
import csv
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import mplcyberpunk
plt.style.use("cyberpunk")
# Ruta del archivo CSV de prueba
coco_csv_path_test = "/Users/nicole/Documents/2023-1/PFC2/Data Set/emotions/test.csv"

# Obtener las imágenes y las etiquetas del archivo CSV de prueba
images_ = []
emotions_ = []
ages_ = []

max_images_per_label = 500

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
# Ruta de la carpeta que contiene las imágenes
images_folder = "/Users/nicole/Documents/2023-1/PFC2/Data Set/emotions"

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
        if age_counts_[label]["no kid"] < 0 * max_images_per_label:
            age_counts_[label]["no kid"] += 1
        else:
            continue

    # Ruta completa de la imagen
    image_file = os.path.join(images_folder, str(image_info["path"]))

    # Leer la imagen
    image = cv2.imread(image_file)
    # Redimensionar a 48x48 píxeles y convertir a escala de grises
    image = cv2.resize(image, (144, 144))

    # Incrementar el contador de imágenes por etiqueta
    label_counts_[label] += 1

    X_.append(image)
    y_.append(label)

# Realizar el conteo de elementos para cada etiqueta
for emotion, count in label_counts_.items():
    print(f"{emotion}: {count} elementos (niños: {age_counts_[emotion]['kid']}, no niños: {age_counts_[emotion]['no kid']})")


# Convertir las listas a matrices numpy
X_ = np.array(X_)
y_ = np.array(y_)

# Carga el modelo FER
model = FER(mtcnn=True)

# Lista para almacenar las predicciones
predictions = []

# Loop a través de las imágenes
for image in X_:
    # Detecta las caras y predice las emociones
    result = model.detect_emotions(image)

    # Verifica si se detectaron caras en la imagen
    if result is not None and len(result) > 0:
        # Obtiene la emoción con mayor probabilidad para la primera cara detectada
        emotion = result[0]["emotions"]
        predicted_emotion = max(emotion, key=emotion.get)
        predictions.append(predicted_emotion)
    else:
        predictions.append(None)  # Si no se detecta una cara en la imagen, agregar None como predicción


# Elimina las predicciones None y las etiquetas correspondientes
valid_predictions = [pred for pred in predictions if pred is not None]
valid_labels = [label for pred, label in zip(predictions, y_) if pred is not None]

# Inicializar el codificador de etiquetas
label_encoder = LabelEncoder()

# Codificar las etiquetas originales
y_encoded = label_encoder.fit_transform(valid_labels)

# Obtener los nombres reales de las etiquetas
label_names = np.unique(valid_labels)

print(classification_report(valid_labels, valid_predictions))

# Calcula la precisión (accuracy) y puntuación F1
accuracy = accuracy_score(valid_labels, valid_predictions)
f1 = f1_score(valid_labels, valid_predictions, average="weighted")

print("Accuracy:", accuracy)
print("F1 Score:", f1)

# Generar la matriz de confusión
confusion_mtx = confusion_matrix(valid_labels, valid_predictions)
print(confusion_mtx)

# Crear el heatmap de la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mtx, annot=True, fmt='d')
mplcyberpunk.add_glow_effects()

# Personalizar etiquetas y títulos
plt.xlabel('Predicted')
plt.ylabel('True')

x_ticks = np.arange(len(label_names))

# Reemplazar los valores en los ejes x e y por los nombres de las etiquetas reales
plt.xticks(x_ticks + 0.5, label_names, rotation=90)
plt.yticks(x_ticks + 0.5, label_names, rotation=0)

plt.show()
