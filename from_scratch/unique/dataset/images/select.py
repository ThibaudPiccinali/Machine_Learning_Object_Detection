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



# dossier = 'input'  
# extension = ".xml"  

# # Parcours de tous les fichiers du dossier
# for nom_fichier in os.listdir(dossier):
#     if nom_fichier.endswith(extension):
#         # Construction du chemin complet du fichier
#         chemin_fichier = os.path.join(dossier, nom_fichier)
#         # Suppression du fichier
#         os.remove(chemin_fichier)
#         print("Fichier supprimé :", nom_fichier)

for i in range(1,10):
    t=1
    while(t!=0):
        t=1
        n=random.randint(1,5000)
        g="/"+str(n)+".png"
        xml="/"+str(n)+".xml"
        if (os.path.exists(file_source + g)):
            im = cv2.imread(file_source + g)
            print(im.shape)
            if(im.shape == (1024,1024,3)):
                t=0
                shutil.move(file_source + g, file_destination)
                shutil.move(file_source + xml, file_destination)