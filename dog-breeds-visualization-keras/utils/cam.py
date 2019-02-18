# Source: https://github.com/alexisbcook/ResNetCAM-keras

import numpy as np
import scipy
from keras.models import Model


def CAM(img, model, pred):
    # get AMP layer weights
    all_amp_layer_weights = model.layers[-1].get_weights()[0]
    # extract activation output
    cam_model = Model(inputs=model.input, outputs=model.layers[-3].output)
    last_conv_output = cam_model.predict(img)

    # change dimensions of last convolutional output to 8 x 8 x 2048
    last_conv_output = np.squeeze(last_conv_output)
    # bilinear upsampling to resize each filtered image to size of original image
    # ...may take up to a minute
    mat_for_mult = scipy.ndimage.zoom(last_conv_output, (img.shape[1] / 8, img.shape[2] / 8, 1), order=1)
    # get AMP layer weights for the passed prediction class
    amp_layer_weights = all_amp_layer_weights[:, pred]
    # get class activation map
    final_output = np.dot(mat_for_mult.reshape((img.shape[1]*img.shape[2], 2048)),
                          amp_layer_weights).reshape(img.shape[1], img.shape[2])
    return final_output
