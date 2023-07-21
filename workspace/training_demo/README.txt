	# 1 Dans un premier temps il faut convertir les fichiers .xml en . record. Pour cela on exécute le code suivant (attention à bien modifier le nom des chemins d'accès) dans le dossier 'TensorFlow/scripts/preprocessing' : 

python generate_tfrecord.py -x C:/Users/islab/Desktop/Tensorflow/workspace/training_demo/images/focal_point/train -l C:/Users/islab/Desktop/Tensorflow/workspace/training_demo/annotations/focal_point/label_map.pbtxt -o C:/Users/islab/Desktop/Tensorflow/workspace/training_demo/annotations/focal_point/train.record

python generate_tfrecord.py -x C:/Users/islab/Desktop/Tensorflow/workspace/training_demo/images/focal_point/test -l C:/Users/islab/Desktop/Tensorflow/workspace/training_demo/annotations/focal_point/label_map.pbtxt -o C:/Users/islab/Desktop/Tensorflow/workspace/training_demo/annotations/focal_point/test.record

	# 2 Il faut ensuite mettre à jour le fichier 'pipeline.config' dans le dossier 'models'

	# 3 Il ne reste plus qu'à entrainer le modèle (on exécute dans le dossier 'training_demo'). Faire bien attention à modifier le nom du modèle si différent !

python model_main_tf2.py --model_dir=models/efficientdet_d0_coco17_tpu-32_focal_point --pipeline_config_path=models/efficientdet_d0_coco17_tpu-32_focal_point/pipeline.config

	# 4 Optionel : Si on veut observer le modèle s'entrainer :

tensorboard --logdir=models/efficientdet_d0_coco17_tpu-32_focal_point (exécuter sur un nouveau terminal dans 'training_demo')
 
Ouvrir dans un navigateur 'http://localhost:6006/'

	# 5 Exporter le modèle (exécuter dans 'training_demo') :

python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\models\efficientdet_d0_coco17_tpu-32_focal_point\pipeline.config --trained_checkpoint_dir .\models\efficientdet_d0_coco17_tpu-32_focal_point --output_directory .\exported-models\my_model_focal_point