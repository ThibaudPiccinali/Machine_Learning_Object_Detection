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
input_folder = "usemodel/input"

# Dossier de sortie pour les images modifiées
output_folder = "usemodel/output"

# Définir les paramètres de prétraitement
img_width = 1024
img_height = 1024

# Charger le modèle
model = load_model('model.h5')

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

    # Création de l'objet Figure et de l'axe avec la taille de l'image
    fig, ax = plt.subplots(figsize=(img_width / 100, img_height / 100))

    # Affichage de l'image
    ax.imshow(image)


    # Récupérer les coordonnées des bounding box
    c=0
    for i in range(0,len(pred[0])//4):
        x1=pred[0][c]
        y1=pred[0][c+1]
        x2=pred[0][c+2] 
        y2=pred[0][c+3]
        if(x1>0 and y1>0 and x2>0 and y2>0):
            # Dessiner les boîtes englobantes pour chaque focal point détecté
            rect = patches.Rectangle((int(x1), int(y1)), int(x2) - int(x1), int(y2) - int(y1), linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
        c=c+4

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












