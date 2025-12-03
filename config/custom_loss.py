import tensorflow as tf
from keras.saving import register_keras_serializable

@register_keras_serializable()
def custom_mse(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()
    return mse(y_true, y_pred)
