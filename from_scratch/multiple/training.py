import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Reshape


# Fonction(s) utile(s)

def plot_scores(train):
    accuracy = train.history['accuracy']
    val_accuracy = train.history['val_accuracy']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'b', label='Score apprentissage')
    plt.plot(epochs, val_accuracy, 'r', label='Score validation')
    plt.title('Scores')
    plt.legend()
    # Enregistrement de l'image modifiée
    plt.savefig("stats_model")
    plt.show()

# Définir les paramètres de prétraitement
img_width = 676
img_height = 380
nbr_focal_point_max = 7

# Définir les paramètres de l'entraînement
batch_size = 32
epochs = 32

# Définir les chemins d'accès aux données
data_dir = 'dataset/images'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'test')

# Fonction pour charger les images et les annotations au format XML
def load_data_from_xml(xml_path, img_width, img_height, max_focal_point):
    # Parser le fichier XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Récupérer les annotations
    annotations = []
    count = 0
    for obj in root.findall('object'):
        if count >= max_focal_point:
            break

        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        annotations.append([1,xmin, ymin, xmax, ymax])
        count += 1

    # Charger l'image
    img_path = os.path.join(os.path.dirname(xml_path), root.find('filename').text)
    img = cv2.imread(img_path)

    # Redimensionner l'image
    img = cv2.resize(img, (img_width, img_height))

    # Normaliser l'image
    img = img / 255.0

    # Compléter avec des coordonnées vides si moins de focal point détecté
    while count < max_focal_point:
        annotations.append([0, 0, 0, 0, 0])
        count += 1
    return img, annotations


# Fonction pour charger les données
def load_data(folder, img_width, img_height):
    # Récupérer les chemins d'accès aux fichiers XML
    xml_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.xml')]
    # Charger les données
    images = []
    annotations = []
    for xml_file in xml_files:
        img, annot = load_data_from_xml(xml_file, img_width, img_height, nbr_focal_point_max)
        images.append(img)
        annotations.append(annot)
    print( np.array([np.array(x) for x in annotations]))
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
    Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),

    # Deuxième couche de convolution
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # Deuxième couche de convolution
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

        # Deuxième couche de convolution
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

            # Deuxième couche de convolution
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # Couche de mise à plat
    Flatten(),

    # Couche entièrement connectée
    Dense(512, activation='relu'),

    # Couche de sortie
    Dense(nbr_focal_point_max * 5, activation='linear'),  # nbr_focal_point_max * 4 coordonnées (xmin, ymin, xmax, ymax)


])

# Compiler le modèle
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Afficher un résumé de l'architecture du modèle
model.summary()

# Entraînement du modèle
train = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(val_images, val_labels))

## Sauvegarde du modèle (pour de futur utilisation)
model.save("model.h5")

# Montre les résultats
plot_scores(train)
