import os
import pickle
import numpy as np
import time
import argparse
import shutil
from sklearn.metrics import confusion_matrix

from keras.models import load_model, Model
from keras.layers import Input, Dense, Dropout
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.utils import to_categorical
from keras import optimizers

# Global constants
DIR = os.path.dirname(os.path.realpath(__file__))  # current path
DATA_DIR = os.path.join(DIR, 'data')  # path to data
TEMP_DIR = os.path.join(DIR, 'temp')  # path to temporary folder
WORK_DIR = os.path.join(TEMP_DIR, str(int(time.time())))  # path to working folder
os.makedirs(WORK_DIR, exist_ok=True)

IMG_SIZE = 299


def train_from_features(config):
    """
    ConvNet as fixed feature extractor

    - Using the bottleneck features of a pre-trained network
    """

    # Load the bottleneck features and the corresponding labels
    train_data = np.load(os.path.join(TEMP_DIR, config.features_dir + '/train_data.npy'))
    train_labels = np.load(os.path.join(TEMP_DIR, config.features_dir + '/train_labels.npy'))
    val_data = np.load(os.path.join(TEMP_DIR, config.features_dir + '/val_data.npy'))
    val_labels = np.load(os.path.join(TEMP_DIR, config.features_dir + '/val_labels.npy'))

    # Train a small FC model from scratch
    if config.load_model is not None:
        top_model = load_model(os.path.join(TEMP_DIR, config.load_model))
    else:
        input = Input(shape=train_data.shape[1:])
        output = Dense(len(set(train_labels)), activation='softmax')(input)
        top_model = Model(inputs=input, outputs=output)

    # Callbacks
    # Checkpointing the best model & restoring that as our model for prediction
    cb_checkpointer = ModelCheckpoint(
        filepath=os.path.join(WORK_DIR, 'model_fc.hdf5'),
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

    # One-hot encoding
    train_classes = to_categorical(train_labels, dtype='int32')
    val_classes = to_categorical(val_labels, dtype='int32')

    # Fit the model to data
    top_model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(),
                  metrics=['accuracy'])
    fit_history = top_model.fit(
        train_data, train_classes,
        validation_data=(val_data, val_classes),
        epochs=config.epochs,
        callbacks=[cb_checkpointer, cb_lr, cb_tensorboard])
    
    # Save the full model to the disk
    if config.save_full:
        base_model = InceptionV3(include_top=False, pooling='avg')
        top_model = load_model(os.path.join(WORK_DIR, 'model_fc.hdf5'))
        model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
        model.save(os.path.join(WORK_DIR, 'model.hdf5'))
        os.remove(os.path.join(WORK_DIR, 'model_fc.hdf5'))

    # Save the metrics history to the disk
    with open(os.path.join(WORK_DIR, 'fit_history.history'), 'wb') as f:
        pickle.dump(fit_history.history, f)

    # Save normalized confusion matrix for later use in penalization
    if config.save_conf_weights:
        pred = model.predict(val_data)
        conf_weights = confusion_matrix(val_labels, pred.argmax(axis=1))
        np.fill_diagonal(conf_weights, 0)
        conf_weights = 1 + conf_weights / np.max(conf_weights)
        np.save(os.path.join(WORK_DIR, 'conf_weights.npy'), conf_weights)
        
    # Rename dir to contain loss info
    best_loss = np.round(np.min(fit_history.history['val_loss']), 4)
    NEW_DIR = os.path.join(TEMP_DIR, 'loss_' + str(best_loss))
    if os.path.isdir(NEW_DIR):
        shutil.rmtree(NEW_DIR)
    os.rename(WORK_DIR, NEW_DIR)
    print("Best model:", NEW_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_dir', type=str, default='features_' + str(IMG_SIZE))
    parser.add_argument('--load_model', type=str)
    parser.add_argument('--save_full', action="store_true", default=False)
    parser.add_argument('--save_conf_weights', action="store_true", default=False)
    parser.add_argument('--epochs', type=int, required=True)
    config = parser.parse_args()
    print('Config:', config)

    train_from_features(config)
