################# Ce code permet de faire une prediction sur l'ensemble des images du dossier "PATH_TO_TEST_DIR" grâce au modèle présent dans "PATH_TO_MODEL_DIR"

# ~~~~~~~~
# Imports
# ~~~~~~~~

import os
import pathlib
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

# ~~~~~~~~~~~~~~~~~~~~~~ 
# Parametres à modifier 
# ~~~~~~~~~~~~~~~~~~~~~~

PATH_TO_MODEL_DIR="exported-models\my_model_focal_point" # Le dossier qui contient le modèle entrainé
PATH_TO_TEST_DIR="images/focal_point/predictions/input/" # Le dossier qui contient les images dont on veut faire la prediction
PATH_TO_SAVE="images/focal_point/predictions/output/" # Le dossier dans lequel se sauvegarde les images avec la prediction
PATH_TO_LABELS="annotations/focal_point/label_map.pbtxt" # Le fichier qui contient les labels du dataset

# ~~~~~~~~~~~~~~~
# Load the model
# ~~~~~~~~~~~~~~~

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load label map data (for plotting)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Putting everything together
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))

IMAGE_PATHS=[]

for filename in os.listdir(PATH_TO_TEST_DIR):
   if filename.endswith(".png"):
        IMAGE_PATHS.append(PATH_TO_TEST_DIR + filename)


i=0

for image_path in IMAGE_PATHS:
    print('Running inference for {}... '.format(image_path), end='')

    image_np = load_image_into_numpy_array(image_path)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)


    plt.figure()

    plt.imshow(image_np_with_detections)
            # Enregistrement de l'image modifiée
    plt.savefig(PATH_TO_SAVE+str(i)+".png")
    print('Done')
    i+=1
