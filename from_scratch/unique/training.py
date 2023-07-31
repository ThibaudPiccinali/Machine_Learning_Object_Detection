
import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

# Fonction(s) utile(s)

def plot_scores(train) :
    accuracy = train.history['accuracy']
    val_accuracy = train.history['val_accuracy']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'b', label='Score apprentissage')
    plt.plot(epochs, val_accuracy, 'r', label='Score validation')
    plt.title('Scores')
    plt.legend()
    # Enregistrement de l'image modifiée
    plt.savefig("stats_model_landscape_object")
    plt.show()

# Définir les paramètres de prétraitement
img_width = 1024
img_height = 1024

# Définir les paramètres de l'entraînement
batch_size = 24
epochs = 15

# Définir les chemins d'accès aux données
data_dir = 'dataset_landscape_object/images'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'test')

# Fonction pour charger les images et les annotations au format XML
def load_data_from_xml(xml_path, img_width, img_height):
    # Parser le fichier XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Récupérer les annotations
    annotations = []
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        annotations.append([xmin, ymin, xmax, ymax])

    # Charger l'image
    img_path = os.path.join(os.path.dirname(xml_path), root.find('filename').text)
    img = cv2.imread(img_path)
    
    # Redimensionner l'image
    img = cv2.resize(img, (img_width, img_height))
    print(root.find('filename').text)
    print(annotations)
    # Normaliser l'image
    img = img / 255.0
    return img, annotations

# Fonction pour charger les données
def load_data(folder, img_width, img_height):
    # Récupérer les chemins d'accès aux fichiers XML
    xml_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.xml')]
    # Charger les données
    images = []
    annotations = []
    for xml_file in xml_files:
        img, annot = load_data_from_xml(xml_file, img_width, img_height)
        images.append(img)
        annotations.append(annot) 
    return np.array(images), np.array([np.array(x) for x in annotations])

# Charger les données d'entraînement
train_images, train_labels = load_data(train_dir, img_width, img_height)
# Charger les données de validation
val_images, val_labels = load_data(val_dir, img_width, img_height)

train_labels = train_labels.reshape(train_labels.shape[0], -1)
val_labels = val_labels.reshape(val_labels.shape[0], -1)
train_labels = np.asarray(train_labels).astype(np.float32)
val_labels = np.asarray(val_labels).astype(np.float32)

# Définition de l'architecture du modèle

model = Sequential([ 
    # Première couche de convolution
    Conv2D(64, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),

    # Deuxième couche de convolution
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),


    # Couche de mise à plat
    Flatten(),

    # Couche entièrement connectée
    Dense(128, activation='relu'),

    # Couche de sortie
    Dense(4, activation='linear')
])

# Compiler le modèle
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Afficher un résumé de l'architecture du modèle
model.summary()

# Entraînement du modèle
train = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(val_images, val_labels))

## Sauvegarde du modèle (pour de futur utilisation)
model.save("model_landscape_object.h5")

# Montre les résultats
plot_scores(train)