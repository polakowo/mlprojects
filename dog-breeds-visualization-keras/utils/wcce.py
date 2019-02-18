import itertools
import keras.backend as K


class WeightedCategoricalCrossEntropy(object):
    """
    Penalizes misclassification between pre-defined (target, output) pairs

    https://github.com/keras-team/keras/issues/2115
    """

    def __init__(self, weights):
        # On mnist we put a higher cost when a 1 is missclassified as a 7 and vice versa:
        # w_array = np.ones((10, 10))
        # w_array[1, 7] = 1.2
        # w_array[7, 1] = 1.2
        self.weights = weights
        self.__name__ = 'w_categorical_crossentropy'

    def __call__(self, y_true, y_pred):
        return self.w_categorical_crossentropy(y_true, y_pred)

    def w_categorical_crossentropy(self, y_true, y_pred):
        nb_cl = len(self.weights)
        final_mask = K.zeros_like(y_pred[..., 0])
        y_pred_max = K.max(y_pred, axis=-1)
        y_pred_max = K.expand_dims(y_pred_max, axis=-1)
        y_pred_max_mat = K.equal(y_pred, y_pred_max)
        for c_p, c_t in itertools.product(range(nb_cl), range(nb_cl)):
            w = K.cast(self.weights[c_t, c_p], K.floatx())
            y_p = K.cast(y_pred_max_mat[..., c_p], K.floatx())
            y_t = K.cast(y_pred_max_mat[..., c_t], K.floatx())
            final_mask += w * y_p * y_t
        return K.categorical_crossentropy(y_true, y_pred) * final_mask
