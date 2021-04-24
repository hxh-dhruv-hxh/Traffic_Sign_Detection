import matplotlib
matplotlib.use("agg")

# Loading the saved model
import time
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os



# Label map path
PATH_TO_LABELS = "records/classes.pbtxt"

# Saved model path
PATH_TO_SAVED_MODEL = "exported-models/saved_model/"

print("[INFO] Loading model...", end="")
start_time = time.time()

# Loading the saved model and building the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# Setting category index
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
print(category_index)
# Seting up images for inference
# Using the images in the test directory
IMAGE_DIR = "test_images/"
IMAGE_PATHS = []

for file in os.listdir(IMAGE_DIR):

    if file.endswith(".jpg") or file.endswith(".png"):
        IMAGE_PATHS.append(os.path.join(IMAGE_DIR, file))

#print(IMAGE_PATHS)

# Making prediction

for image_path in IMAGE_PATHS:

    print("Running inference for {}...".format(image_path), end='')

    # Changing the image into numpy array to feed into tensorflow graph, by convention we put it into numpy array with shape
    # ... (height, width, channels) where channels = 3 for RGB
    image_np = np.array(Image.open(image_path))

    # the input needs to be a tensor
    input_tensor = tf.convert_to_tensor(image_np)

    # Model inputs a batch of images, so we need to add an axis with tf.newaxis
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)
    #print(detections)
    # All the batches are tensors, so we convert them into numpy arrays and take index [0] to remove the batch dimension
    # ...we are only interested in the first num_detections
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # Detection classes should be ints
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(image_np_with_detections,
                                                        detections['detection_boxes'],
                                                        detections['detection_classes'],
                                                        detections['detection_scores'],
                                                        category_index,
                                                        use_normalized_coordinates=True,
                                                        max_boxes_to_draw=20,
                                                        min_score_thresh=0.50,
                                                        agnostic_mode=False)
    # plt.figure(figsize=(12, 8))
    # plt.imshow(image_np_with_detections)
    # print('Done')
    #print("Done")
    # cv2.imshow('image', image_np_with_detections)
    # cv2.waitKey(0)
    # plt.show()
    img = Image.fromarray(image_np_with_detections, 'RGB')
    img.show()

plt.show()









































