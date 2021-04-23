import os

# Initialize the base path of the lisa dataset
BASE_PATH = 'dataset/LISA'

# Building the path to the annotations file
ANNOT_PATH = os.path.sep.join([BASE_PATH, "allAnnotations.csv"])

# Building the path to the output training and testing record files along with the class label files
TRAIN_RECORD = os.path.sep.join([BASE_PATH, "records/training.record"])
TEST_RECORD = os.path.sep.join([BASE_PATH, 'records/testing.record'])
CLASSES_FILE = os.path.sep.join([BASE_PATH, 'records/classes.pbtxt'])

# Setting our test size variable
TEST_SIZE = 0.25

# Initializing the class labels directory
CLASSES = {'pedestrianCrossing': 1, 'signalAhead': 2, "stop": 3}

