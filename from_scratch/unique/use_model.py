# Version avec MatPlotlib

from ast import For
import os
from turtle import width
import xml.etree.ElementTree as ET
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential, load_model
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import matplotlib.image as mpimg

# Dossier contenant les images d'entrée
input_folder = "usemodel_landscape_object/input"

# Dossier de sortie pour les images modifiées
output_folder = "usemodel_landscape_object/output"

# Définir les paramètres de prétraitement
img_width = 1024
img_height = 1024

# Charger le modèle
model = load_model('model_landscape_object.h5')

# Parcours des fichiers du dossier d'entrée
for filename in os.listdir(input_folder):
    # Chemin complet vers l'image d'entrée
    input_path = os.path.join(input_folder, filename)
        
    # Chargement de l'image
    image = Image.open(input_path)

    # Normaliser l'image
    image = np.array(image) / 255.0

    # Faire la prédiction
    pred = model.predict(np.array([image]))
    print(pred)

    # Récupérer les coordonnées de la bounding box
    x1, y1, x2, y2 = pred[0]

    # Création de l'objet Figure et de l'axe avec la taille de l'image
    fig, ax = plt.subplots(figsize=(1024 / 100, 1024 / 100))

    # Affichage de l'image
    ax.imshow(image)

    # Créer un rectangle pour représenter la boîte
    rect = patches.Rectangle((int(x1), int(y1)), int(x2) - int(x1), int(y2) - int(y1), linewidth=2, edgecolor='g', facecolor='none') 
    # Ajouter le rectangle à l'axe
    ax.add_patch(rect)

    # Afficher l'axe sans les graduations
    plt.axis('off')

    # Réglage des limites de l'axe pour correspondre à l'image
    ax.set_xlim(0, 1024)
    ax.set_ylim(1024, 0)

    # Suppression des espaces blancs autour de l'image
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Chemin complet vers l'image de sortie
    output_path = os.path.join(output_folder, filename)

    # Enregistrement de l'image modifiée
    plt.savefig(output_path)

    # Fermeture de la figure
    plt.close()












