import random
import shutil
import os
import cv2
import numpy as np
from PIL import Image

file_source = 'test'
file_destination = 'train'

dossier = "input"  
nouvelle_taille = (1024, 1024)  

# # Parcours de tous les fichiers du dossier
# for nom_fichier in os.listdir(dossier):
#     chemin_fichier = os.path.join(dossier, nom_fichier)
#     if os.path.isfile(chemin_fichier):
#         try:
#             # Ouverture de l'image
#             image = Image.open(chemin_fichier)
#             # Redimensionnement de l'image
#             image_resized = image.resize(nouvelle_taille)
#             # Sauvegarde de l'image redimensionnée (écrase l'originale)
#             image_resized.save(chemin_fichier)
#             print("Image redimensionnée :", nom_fichier)
#         except:
#             print("Erreur lors du traitement de l'image :", nom_fichier)



dossier = 'train'  
extension = ".Identifier"  

# # Parcours de tous les fichiers du dossier
for nom_fichier in os.listdir(dossier):
    if nom_fichier.endswith(extension):
#         # Construction du chemin complet du fichier
        chemin_fichier = os.path.join(dossier, nom_fichier)
        # Suppression du fichier
        os.remove(chemin_fichier)
        print("Fichier supprimé :", nom_fichier)

