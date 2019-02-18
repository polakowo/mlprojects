import argparse
import os
import pickle
import numpy as np
import time
import itertools
from collections import Counter

from keras.models import load_model, Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras import optimizers
from utils.wcce import WeightedCategoricalCrossEntropy

# Global constants
DIR = os.path.dirname(os.path.realpath(__file__))  # current path
DATA_DIR = os.path.join(DIR, 'data')  # path to data
TEMP_DIR = os.path.join(DIR, 'temp')  # path to temporary folder
WORK_DIR = os.path.join(TEMP_DIR, str(int(time.time())))  # path to working folder
os.makedirs(WORK_DIR, exist_ok=True)

IMG_SIZE = 299
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VAL = 32


def fine_tune(config):
    """
    Finetuning the convnet

    - *freeze* the bottom N layers and *unfreeze* the remaining top layers
    """
    # Load model from the disk
    model = load_model(os.path.join(TEMP_DIR, config.load_model))

    # Freeze *config.from_layer* layers
    for layer in model.layers[:config.from_layer]:
        layer.trainable = False
    for layer in model.layers[config.from_layer:]:
        layer.trainable = True

    # Load train and validation data into the ImageDataGenerator
    if config.augment:
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=30,
            horizontal_flip=True,
            fill_mode='nearest')
    else:
        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    traingen = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'train'),
        target_size=(config.img_size, config.img_size),
        class_mode='categorical',
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True)
    valgen = val_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'val'),
        target_size=(config.img_size, config.img_size),
        class_mode='categorical',
        batch_size=BATCH_SIZE_VAL,
        shuffle=True)

    # Callbacks
    # Checkpointing the best model & restoring that as our model for prediction
    cb_checkpointer = ModelCheckpoint(
        filepath=os.path.join(WORK_DIR, 'model.hdf5'),
        monitor='val_loss',
        save_best_only=True,
        mode='auto')
    cb_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=1)
    # Saving logs for analysis in TensorBoard
    cb_tensorboard = TensorBoard(
        log_dir=os.path.join(WORK_DIR, 'logs'),
        histogram_freq=0,
        write_graph=True)

    # This is to compensate the imbalanced classes
    if config.use_class_weights:
        def get_class_weights(y):
            counter = Counter(y)
            majority = max(counter.values())
            return {cls: float(majority/count) for cls, count in counter.items()}

        class_weight = get_class_weights(traingen.classes)
    else:
        class_weight = None

    # Penalize misclassified similar images more heavily
    # Note: requires len(conf_weights) ** 2 operations at startup + slows down the optimizer
    if config.load_conf_weights is not None:
        conf_weights = np.load(os.path.join(TEMP_DIR, 'conf_weights.npy'))
        loss = WeightedCategoricalCrossEntropy(conf_weights)
        loss.__name__ = 'w_categorical_crossentropy'
    else:
        loss = 'categorical_crossentropy'

    # Fit the model to generator
    model.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  loss=loss,
                  metrics=['accuracy'])
    fit_history = model.fit_generator(
        traingen,
        steps_per_epoch=traingen.n // BATCH_SIZE_TRAIN,
        epochs=config.epochs,
        validation_data=valgen,
        validation_steps=valgen.n // BATCH_SIZE_VAL,
        class_weight=class_weight,
        callbacks=[cb_checkpointer, cb_lr, cb_tensorboard])

    # Save the metrics history to the disk
    with open(os.path.join(WORK_DIR, 'fit_history.history'), 'wb') as f:
        pickle.dump(fit_history.history, f)
        
    # Rename dir to contain loss info
    best_loss = np.round(np.min(fit_history.history['val_loss']), 4)
    NEW_DIR = os.path.join(TEMP_DIR, 'loss_' + str(best_loss))
    if os.path.isdir(NEW_DIR):
        shutil.rmtree(NEW_DIR)
    os.rename(WORK_DIR, NEW_DIR)
    print("Best model:", NEW_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=IMG_SIZE)
    parser.add_argument('--augment', action="store_true", default=False)
    parser.add_argument('--from_layer', type=int, default=249)
    parser.add_argument('--use_class_weights', action="store_true", default=True)
    parser.add_argument('--load_conf_weights', type=str)
    parser.add_argument('--epochs', type=int, required=True)

    config = parser.parse_args()
    print('Config:', config)

    fine_tune(config)
