# In order to train a network using TFOD API, we need to convert our images and annotations into tensorflow record format, similar to how we
# ...convert image datasets in classification tasks using hdf5.

# importing the necessary packages
from config import lisa_config as config
from pyimagesearch.utils.tfannotation import TFAnnotation
from sklearn.model_selection import train_test_split
from PIL import Image
import tensorflow as tf
import os
import sys


def main(argv=None):

    # Open the classes output file
    f = open(config.CLASSES_FILE, "w")

    # Loop over the classes
    for (k, v) in config.CLASSES.items():

        # Constructing the class information and writing to file
        item = ("item {\n\tid: "+ str(v) + "\n\tname: '"+ k + "'\n}\n")
        f.write(item)

    # Closing the output classes file
    f.close()

    # Initializing the data dictionary which will map each image filename to all bounding boxes associated with the image
    # ... then loading the contents of the annotations file
    D = {}
    rows = open(config.ANNOT_PATH).read().strip().split("\n")

    # Looping over the individual rows and skipping the header
    for row in rows[1:]:

        # Breaking the row into components
        row = row.split(',')[0].split(';')
        (imagePath, label, startX, startY, endX, endY, _) = row
        (startX, startY) = (float(startX), float(startY))
        (endX, endY) = (float(endX), float(endY))

        # If the required label is not in our labels of consideration then ignore it
        if label not in config.CLASSES:
            continue

        # Since, the image can contain multiple traffic signs and therefore multiple bounding boxes, we need to utilize a Python dictionary
        # ... to map the image path (as the key) to a list of labels and associated bounding boxes (the value)

        # Hence, building the path to the input image, then grab any bounding boxes + labels associated with the image path, labels and
        # ... bounding box lists, respectively
        p = os.path.sep.join([config.BASE_PATH, imagePath])
        b = D.get(p, [])

        # Building the tuple consisting of the label and the bounding box, then update the list and store it in the dictionary
        b.append((label, (startX, startY, endX, endY)))

        D[p] = b

    # Creating training and testing splits from data dictionary
    (trainKeys, testKeys) = train_test_split(list(D.keys()), test_size=config.TEST_SIZE, random_state=42)

    # Initialize the data split files
    datasets = [("train", trainKeys, config.TRAIN_RECORD),
                ("test", testKeys, config.TEST_RECORD)]

    # Now, we are ready to build Tensorflow record files
    # Looping over the datasets
    for (dType, keys, outputPath) in datasets:

        # Initializing the tensorflow writer and initialize the total number of examples written to file
        print("[INFO] processing '{}'".format(dType))
        writer = tf.io.TFRecordWriter(outputPath)

        total = 0

        for k in keys:

            # loading the input images from disk as a tensorflow object
            encoded = tf.io.gfile.GFile(k, 'rb').read()
            encoded = bytes(encoded)

            # Loading the image from disk as a PIL object
            pilImage = Image.open(k)
            (w, h) = pilImage.size[:2]

            # Parsing the filename and encoding from the input path
            filename = k.split(os.path.sep)[-1]
            encoding = filename[filename.rfind(".")+1:]

            # Initializing the annotation object used to store information regarding the bounding box+labels
            tfAnnot = TFAnnotation()
            tfAnnot.image = encoded
            tfAnnot.encoding = encoding
            tfAnnot.filename = filename
            tfAnnot.width = w
            tfAnnot.height = h

            # Now, as we have encoded the image, encoding and other info to our tfAnnot object, let's now add bounding box info to annotObj
            for (label, (startX, startY, endX, endY)) in D[k]:

                # Tensorflow assumes all bounding boxes are in the range[0, 1], so we need to scale them
                xMin = startX / w
                xMax = endX / w
                yMin = startY / h
                yMax = endY / h

                # Updating the bounding boxes and labels list
                tfAnnot.xMins.append(xMin)
                tfAnnot.xMaxs.append(xMax)
                tfAnnot.yMins.append(yMin)
                tfAnnot.yMaxs.append(yMax)
                tfAnnot.textLabels.append(label.encode('utf8'))
                tfAnnot.classes.append(config.CLASSES[label])
                tfAnnot.difficult.append(0)

                # Incrementing the total no. of examples written
                total += 1

                # encoding the data point attributes using tensorflow helper functions
                features = tf.train.Features(feature=tfAnnot.build())
                example = tf.train.Example(features=features)

                # Adding the example to the writer
                writer.write(example.SerializeToString())

        # Close the writer and print diagnostic information to the user
        writer.close()
        print("[INFO] {} Examples saved for '{}'".format(total, dType))

# Check to see if the main thread should be started
if __name__ == "__main__":
    tf.compat.v1.app.run()
    
    
