# Tensorflow Object Detection

## Français

Ce dossier contient un ensemble d'éléments qui devraient permettre d'implémenter et d'entraîner un modèle de détection d'objet à partir de votre dataset. L'algorithme final utilisera un modèle pré-entraîné (fourni par *Tensorflow*).

### Prérequis : 

Tout d'abord, il convient de suivre l'ensemble du tutoriel présent sur ce lien : https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html (suivez uniquement la partie *Installation*). Ce dernier permet de bien configurer votre machine et de télécharger toutes les bibliothèques nécessaires. Résumons rapidement l'organisation de vos dossiers (dans le sous-dossier *workspace/training_demo*) que vous devriez avoir après avoir suivi l'ensemble de ce tutoriel, ainsi que leurs utilités respectives (si vous ne comprenez pas tout ce qui est décrit, ce n'est pas grave : chaque élément sera expliqué en temps voulu): 

- Le dossier *annotations* : contiendra un fichier *.PBTXT* qui servira à définir à votre modèle le nom des (différentes) classes. Il comprendra également des fichiers *.record* (équivalent de l'ensemble de vos fichiers *.xml* de votre dataset mais mis à un format lisible pour les bibliothèques de *Tensorflow*).

- Le dossier *images* : contient l'ensemble des images de votre dataset. Il est lui-même réparti en trois sous-dossiers *test*, *train* (pour vos images qui seront utilisées pour l'entraînement et le test de votre modèle) et le dossier *predictions* qui sera utilisé pour stocker les résultats des prédictions de notre modèle. Chaque image des sous-dossiers *test* et *train* doit être labellisée à l'aide de fichier *.xml*.

- Le dossier *pre-trained-models* : regroupe l'ensemble des modèles pré-entraînés téléchargés (sur cette page : https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md).

- Le dossier *model* : comprend l'ensemble des résultats intermédiaires lors de l'entraînement de votre modèle. Vous n'avez pas besoin de vous préoccuper de ce dossier, sachez juste que c'est là où sont stockées les différentes "states" de votre modèle.

- Le dossier *exported-models* : contient votre modèle final, entraîné et prêt à l'utilisation.

### Comment entraîner son premier modèle ?

Le premier modèle que vous allez vouloir entraîner nécessitera plus de préparations que les suivants (annotations des images, téléchargement de modèle pré-entraîné...). Je vous invite donc à suivre ce tutoriel : https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#training-custom-object-detector qui décrit l'ensemble des démarches à suivre.

Si vous avez déjà suivi ce tutoriel au moins une fois et que vous souhaitez juste entraîner un modèle sur un dataset différent, vous pouvez vous contenter de suivre les étapes décrites dans la prochaine partie.

### Comment entraîner un modèle sur un nouveau dataset ?

- 1 Dans un premier temps, il faut convertir les fichiers *.xml* (de votre dataset) en fichiers *.record*. Pour cela, on exécute le code suivant (attention à bien modifier le nom des chemins d'accès : le code ci-dessous est à titre d'exemple) dans le dossier *TensorFlow/scripts/preprocessing* : 

        python generate_tfrecord.py -x C:/Users/islab/Desktop/Tensorflow/workspace/training_demo/images/focal_point/train -l C:/Users/islab/Desktop/Tensorflow/workspace/training_demo/annotations/focal_point/label_map.pbtxt -o C:/Users/islab/Desktop/Tensorflow/workspace/training_demo/annotations/focal_point/train.record

        python generate_tfrecord.py -x C:/Users/islab/Desktop/Tensorflow/workspace/training_demo/images/focal_point/test -l C:/Users/islab/Desktop/Tensorflow/workspace/training_demo/annotations/focal_point/label_map.pbtxt -o C:/Users/islab/Desktop/Tensorflow/workspace/training_demo/annotations/focal_point/test.record

Vous devriez trouver les deux fichiers correspondants dans le dossier *annotations*. Attention, n'oubliez pas de modifier au préalable le fichier *label_map.pbtxt* si nécessaire !

- 2 Il faut ensuite mettre à jour le fichier *pipeline.config* dans le dossier *models*. Pour cela, je vous conseille de reprendre les indications du tutoriel : https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#configure-the-training-pipeline.

- 3 Il ne reste plus qu'à lancer l'entraînement (on exécute dans le dossier *training_demo*). Faites bien attention à modifier le nom d'un modèle si différent !

        python model_main_tf2.py --model_dir=models/efficientdet_d0_coco17_tpu-32_focal_point --pipeline_config_path=models/efficientdet_d0_coco17_tpu-32_focal_point/pipeline.config

- 4 Optionnel : Si vous voulez observer le modèle s'entraîner (exécutez sur un nouveau terminal dans *training_demo*) :

        tensorboard --logdir=models/efficientdet_d0_coco17_tpu-32_focal_point 
 
Puis ouvrez dans un navigateur : 'http://localhost:6006/'

- 5 Enfin, pour exporter le modèle (exécutez dans *training_demo*) :

        python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\models\efficientdet_d0_coco17_tpu-32_focal_point\pipeline.config --trained_checkpoint_dir .\models\efficientdet_d0_coco17_tpu-32_focal_point --output_directory .\exported-models\my_model_focal_point


Attention : le modèle ne termine pas son entraînement de lui-même ! Il faut forcer l'arrêt de ce dernier (CTRL+C, fermer le terminal...). Mais pas de panique : le modèle fait des sauvegardes de manière automatique ! (Ça peut notamment être utile si vous voulez continuer un entraînement plus tard).

### Comment utiliser un modèle (faire des prédictions) ?

Enfin, pour cette dernière partie, constituez deux sous-dossiers dans le dossier *images/predictions*, nommés *input* et *output*. Ces derniers serviront à stocker les images à prédire et les prédictions respectivement.

Il suffit ensuite d'exécuter le code *plot_object_detection_saved_model* en ayant modifié au préalable les variables *PATH_TO_MODEL_DIR*, *PATH_TO_TEST_DIR*, *PATH_TO_SAVE*, *PATH_TO_LABELS*: 
- *PATH_TO_MODEL_DIR* : C'est le dossier qui contient le modèle entrainé,
- *PATH_TO_TEST_DIR* : Celui qui contient les images dont on veut faire la prediction,
- *PATH_TO_SAVE* : Celui dans lequel se sauvegarde les images avec la prediction,
- *PATH_TO_LABELS* : C'est le fichier qui contient les labels du dataset.

Vous devriez trouver dans le dossier *images/predictions/output* l'ensemble de vos résultats.

## English 
This folder contains a set of elements that should allow you to implement and train an object detection model from your dataset. The final algorithm will use a pre-trained model (provided by *Tensorflow*).

### Prerequisites: 

Firstly, you need to follow the entire tutorial available at this link: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html (only follow the *Installation* section). This tutorial will properly configure your machine and download all the necessary libraries. Let us quickly summarize the organization of your folders (inside the *workspace/training_demo* sub-folder) that you should have after following the entire tutorial, along with their respective purposes (if you do not understand everything described, do not worry: each element will be explained in due time):

- The *annotations* folder: It contains a *.PBTXT* file that defines the names of the different classes for your model. It also includes *.record* files, which are equivalent to all your *.xml* files from your dataset but converted to a format readable by *Tensorflow* libraries.

- The *images* folder: This folder holds all the images from your dataset. It is further divided into three sub-folders: *test*, *train* (for images used for training and testing your model), and the *predictions* folder, which will store the results of our model's predictions. Each image in the *test* and *train* sub-folders should be labeled using a *.xml* file.

- The *pre-trained-models* folder: Here, you will find all the downloaded pre-trained models (available on this page: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). 

- The *model* folder: This folder includes all the intermediate results during the training of your model. You do not need to worry about this folder; just know that it is where the different "states" of your model are stored.

- The *exported-models* folder: This folder contains your final model, trained and ready for use.

### How to train your first model?

The first model you want to train will require more preparation than the following ones (annotations of images, downloading pre-trained models, etc.). I invite you to follow this tutorial: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#training-custom-object-detector, which describes all the steps to follow.

If you have already followed this tutorial at least once and you just want to train a model on a different dataset, you can simply follow the steps described in the next section.

### How to train a model on a new dataset?

- Step 1: First, you need to convert the *.xml* files from your dataset to *.record* files. To do this, execute the following code (be sure to modify the path names correctly; the code below is just an example) in the *TensorFlow/scripts/preprocessing* folder: 

        python generate_tfrecord.py -x C:/Users/islab/Desktop/Tensorflow/workspace/training_demo/images/focal_point/train -l C:/Users/islab/Desktop/Tensorflow/workspace/training_demo/annotations/focal_point/label_map.pbtxt -o C:/Users/islab/Desktop/Tensorflow/workspace/training_demo/annotations/focal_point/train.record

        python generate_tfrecord.py -x C:/Users/islab/Desktop/Tensorflow/workspace/training_demo/images/focal_point/test -l C:/Users/islab/Desktop/Tensorflow/workspace/training_demo/annotations/focal_point/label_map.pbtxt -o C:/Users/islab/Desktop/Tensorflow/workspace/training_demo/annotations/focal_point/test.record

You should find the two corresponding files in the annotations folder. Be sure to modify the *label_map.pbtxt* file beforehand if necessary!

- Step 2: Next, update the *pipeline.config* file in the *models* folder. For this, I suggest you follow the instructions from the tutorial: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#configure-the-training-pipeline.

- Step 3: Now, all that is left is to start the training (execute this in the *training_demo* folder). Be careful to modify the model name if different!

        python model_main_tf2.py --model_dir=models/efficientdet_d0_coco17_tpu-32_focal_point --pipeline_config_path=models/efficientdet_d0_coco17_tpu-32_focal_point/pipeline.config

- Step 4 Optional: If you want to observe the model training (execute this in a new terminal in *training_demo*):

        tensorboard --logdir=models/efficientdet_d0_coco17_tpu-32_focal_point 
 
Then open in a browser: 'http://localhost:6006/'

- Step 5: Finally, to export the model (execute this in *training_demo*):

        python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\models\efficientdet_d0_coco17_tpu-32_focal_point\pipeline.config --trained_checkpoint_dir .\models\efficientdet_d0_coco17_tpu-32_focal_point --output_directory .\exported-models\my_model_focal_point


Attention: the model does not end its training on its own! You need to stop it manually (CTRL+C, close the terminal, etc.). But do not worry: the model automatically makes backups, which can be useful if you want to continue training later.

### How to use a model (make predictions)?

Finally, for this last part, create two sub-folders in the *images/predictions* folder, named input and output. These will be used to store the images to predict and the respective predictions.

Then, simply execute the *plot_object_detection_saved_model* code after modifying the variables *PATH_TO_MODEL_DIR*, *PATH_TO_TEST_DIR*, *PATH_TO_SAVE*, *PATH_TO_LABELS*:
- *PATH_TO_MODEL_DIR* : This is the folder containing the trained model.
- *PATH_TO_TEST_DIR* : The folder containing the images you want to make predictions on.
- *PATH_TO_SAVE* : The folder where images with predictions will be saved.
- *PATH_TO_LABELS* : This is the file containing the dataset labels.

You should find all your results in the *images/predictions/output* folder.