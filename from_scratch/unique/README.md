# Machine Learning Object Detection from scratch (unique)

## English

This GitHub repository contains a collection of components that should enable you to implement and train an object detection model from scratch.

It is organized into two sub-folders:

- The *dataset* folder contains all the images from your dataset. It is further divided into two sub-folders *test* and *train* for the images used for training and testing your model. Each image in the *test* and *train* sub-folders must be labeled using *.xml* files. You can perform these annotations using software like *LabelImg*.

- The *usemodel* folder contains directories that will be useful for storing your prediction images.

### Model Training

To train your model, simply run the Python script *training.py*.

In this script, you can (and should!) modify various parameters to make your model fit your dataset as closely as possible (finding the best parameters usually involves extensive testing).

You can adjust the *batch_size* (which is the number of training samples presented to the model before a weight update is performed), the number of *epochs* (where each epoch represents a complete iteration over the training dataset), the size of your images (*img_width* and *img_height*)...
The architecture of your model is at the heart of your program, and you should modify it as well. The architecture chosen in this code is in the form of convolutional layers. To modify your architecture, you can add or remove layers, change activation functions, adjust weights...
Below is an example architecture (present in *training.py*):

        model = Sequential([ 
        # First convolutional layer
        Conv2D(64, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        MaxPooling2D(2, 2),

        # Second convolutional layer
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        # Flatten layer
        Flatten(),

        # Fully connected layer
        Dense(128, activation='relu'),

        # Output layer
        Dense(4, activation='linear')
        ])

The script concludes by saving your model (under the name *model.h5* - you can modify this) and displaying statistics on the training of your model (accuracy of responses as a function of the number of *epochs*).

### Using Your Model (Making Predictions)

To use your model, simply execute the *usemodel.py* code. This will make predictions for all the images present in the *usemodel/input* folder and save the results in the *usemodel/output* folder.

Make sure that the preprocessing size of the images (*img_width* and *img_height*) matches that of your training, and that the model loaded is the one defined by the *training.py* program (*model = load_model('model.h5')* in our example).

## Français

Ce dossier GitHub contient un ensemble d'éléments qui devrait permettre d'implémenter et d'entraîner un modèle de détection d'objet à partir de zéro.

Il est organisé en deux sous-dossiers :

- Le dossier *dataset* contient l'ensemble des images de votre dataset. Il est lui-même divisé en deux sous-dossiers *test* et *train* pour vos images qui seront utilisées pour l'entraînement et le test de votre modèle. Chaque image des sous-dossiers *test* et *train* doit être labellisée à l'aide de fichiers *.xml*. Vous pouvez effectuer ces annotations à l'aide de logiciels comme *LabelImg*.

- Le dossier *usemodel* contient des dossiers qui seront utiles pour stocker les images de vos prédictions.

### Entraînement du modèle

Pour entraîner votre modèle, il suffit d'exécuter le script Python *training.py*.

Dans ce dernier, vous pouvez (et devez !) modifier différents paramètres pour que votre modèle convienne le plus possible à votre dataset (pour trouver ces paramètres, le meilleur choix reste de faire beaucoup de tests). 

Vous pouvez modifier le nombre *batch_size* (le batch size correspond au nombre d'échantillons d'entraînement présentés au modèle avant qu'une mise à jour des poids soit effectuée), le nombre d'*epochs* (une epoch est une seule itération complète de l'ensemble de données d'entraînement pendant la phase d'apprentissage), la taille de vos images (*img_width* et *img_height*)...
L'architecture de votre modèle est au cœur de votre programme. Vous devez également la modifier. L'architecture choisie dans ce code est sous la forme de couches de convolution. Pour modifier votre architecture, vous pouvez ajouter, supprimer des couches, changer les fonctions d'activation, les poids...
Ci-dessous se trouve un exemple d'architecture (présent dans *training.py*) :

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

Le script se termine en enregistrant votre modèle (sous le nom *model.h5* - vous pouvez le modifier) et en vous affichant des statistiques sur l'entraînement de votre modèle (précision des réponses en fonction du nombre d'*epochs*).

### Utilisation de votre modèle (faire des prédictions)

Pour utiliser votre modèle, vous devez simplement exécuter le code *usemodel.py*. Ce dernier se chargera de faire des prédictions pour toutes les images présentes dans le dossier *usemodel/input* et de les enregistrer dans le dossier *usemodel/output*.

Assurez-vous simplement que la taille de prétraitement des images (*img_width* et *img_height*) correspond à celle de votre entraînement et que le modèle qui est chargé est celui défini par le programme *training.py* (*model = load_model('model.h5')* dans notre exemple).