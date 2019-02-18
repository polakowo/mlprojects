# For further details see:
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

import os
import numpy as np
import shutil
import argparse

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

# Global constants
DIR = os.path.dirname(os.path.realpath(__file__))  # current path
DATA_DIR = os.path.join(DIR, 'data')  # path to data
TEMP_DIR = os.path.join(DIR, 'temp')  # path to temporary folder

IMG_SIZE = 299


def extract_features(config):
    """
    Extract bottleneck features from a pre-trained network
    """
    # Load the pre-trainable Xception model
    base_model = InceptionV3(include_top=False, pooling='avg')

    # Training data
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    traingen = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'train'),
        target_size=(config.img_size, config.img_size),
        batch_size=1,
        class_mode=None,  # generator will only yield batches of data, no labels
        shuffle=False)  # data will be in order
    train_data = base_model.predict_generator(
        traingen,
        steps=traingen.n,
        verbose=1)
    train_labels = traingen.classes

    # Validation data
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    valgen = val_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'val'),
        target_size=(config.img_size, config.img_size),
        batch_size=1,
        class_mode=None,
        shuffle=False)
    val_data = base_model.predict_generator(
        valgen,
        steps=valgen.n,
        verbose=1)
    val_labels = valgen.classes

    # Save the outputs as Numpy arrays
    features_dir = os.path.join(TEMP_DIR, 'features_' + str(config.img_size))
    os.makedirs(features_dir, exist_ok=True)
    np.save(features_dir + '/train_data.npy', train_data)
    np.save(features_dir + '/train_labels.npy', train_labels)
    np.save(features_dir + '/val_data.npy', val_data)
    np.save(features_dir + '/val_labels.npy', val_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=IMG_SIZE)
    config = parser.parse_args()
    print('Config:', config)

    extract_features(config)
