# Machine Learning Object Detection

## Français

Ce dépôt Github contient un ensemble d'éléments qui devrait permettre d'implémenter et d'entraîner un modèle de détection d'objet, à partir de votre dataset.

### Prérequis : 

Suivre l'ensemble de ce tutoriel (https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html) qui permet de bien setup votre machine et de télécharger les bibliothèques nécessaires. Le tutoriel explique également comment télécharger un modèle pré-entrainé et de l'entrainer sur votre dataset. Résumons rapidement l'organisation de vos dossiers (dans le sous dossier *workspace/training_demo*) : 

- le dossier *annotations* : contient un fichier *.PBTXT* qui sert à définir à votre modèle le nom des différentes classes.

- le dossier *images* : contient l'ensemble des de votre dataset. Il est lui même réparti en trois sous dossier *test*, *train* (pour vos images qui seront utilisée pour l'entraînement et le test de votre modèle) et le dossier *predictions* qui sera utilisé pour stocker le résultats des prédictions de notre modèle

- le dossier *pre-trained-models* : regroupe l'ensemble des modèles pré-entrainé téléchargé (sur cette page : https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) 

- le dossier *model* : comprend l'ensemble des résultats intermédiaire lors de l'entraînement de votre modèle

- le dossier *exported-models* : continent votre modèle final, entraîné et prêt à l'utilisation.


### Comment entraîner un modèle sur un nouveau dataset ?

- 1: Dans un premier temps il faut convertir les fichiers *.xml* (de votre dataset) en fichier *.record*. Pour cela on exécute le code suivant (attention à bien modifier le nom des chemins d'accès) dans le dossier *TensorFlow/scripts/preprocessing* : 

        python generate_tfrecord.py -x C:/Users/islab/Desktop/Tensorflow/workspace/training_demo/images/focal_point/train -l C:/Users/islab/Desktop/Tensorflow/workspace/training_demo/annotations/focal_point/label_map.pbtxt -o C:/Users/islab/Desktop/Tensorflow/workspace/training_demo/annotations/focal_point/train.record

        python generate_tfrecord.py -x C:/Users/islab/Desktop/Tensorflow/workspace/training_demo/images/focal_point/test -l C:/Users/islab/Desktop/Tensorflow/workspace/training_demo/annotations/focal_point/label_map.pbtxt -o C:/Users/islab/Desktop/Tensorflow/workspace/training_demo/annotations/focal_point/test.record

- 2 Il faut ensuite mettre à jour le fichier *pipeline.config* dans le dossier *models* (suivre les indications du tutoriel https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#configure-the-training-pipeline ) : 

- 3 Il ne reste plus qu'à lancer l'entraînement (on exécute dans le dossier *training_demo*). Faire bien attention à modifier le nom d'un modèle si différent !

        python model_main_tf2.py --model_dir=models/efficientdet_d0_coco17_tpu-32_focal_point --pipeline_config_path=models/efficientdet_d0_coco17_tpu-32_focal_point/pipeline.config

- 4 Optionnel : Si on veut observer le modèle s'entraîner (exécuté sur un nouveau terminal dans *training_demo*) :

        tensorboard --logdir=models/efficientdet_d0_coco17_tpu-32_focal_point 
 
Puis ouvrir dans un navigateur  : 'http://localhost:6006/'

- 5 Enfin pour exporter le modèle (exécuté dans *training_demo*) :

        python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\models\efficientdet_d0_coco17_tpu-32_focal_point\pipeline.config --trained_checkpoint_dir .\models\efficientdet_d0_coco17_tpu-32_focal_point --output_directory .\exported-models\my_model_focal_point


Attention : le modèle ne termine pas son entrainement de lui-même ! Il faut forcer l'arrêt de ce dernier (CTRL+C, fermer le terminal...). Mais pas de panique: le modèle fait des sauvegarde de manière automatique ! (ça peut notamment être utile si vous voulez continuer un entraînement plus tard).

### Comment utiliser un modèle (faire des prédictions)

Pour cela, constituez deux sous-dossiers dans le dossier *images/prédictions*, nommé *input* et *output*. Ces derniers serviront à stocker les images à prédires et les prédictions respectivement.

Il suffit ensuite d'exécuter le code *plot_object_detection_saved_model* en ayant modifié au préalable les variables *PATH_TO_MODEL_DIR*, *PATH_TO_TEST_DIR*, *PATH_TO_SAVE*, *PATH_TO_LABELS* (se référer au code pour plus de précision). 

Vous devriez trouver dans le dossier images/predictions/output l'ensemble de vos résultats.

## English
This GitHub requisitory contains files and elements to implement your own object detection model.