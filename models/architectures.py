import tensorflow as tf
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from config.custom_loss import custom_mse

def modelo_simple(window):
    model = Sequential([
        tf.keras.layers.Input(shape=(window, 1)),
        GRU(64),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss=custom_mse)
    return model

def modelo_avanzado(window):
    model = Sequential([
        tf.keras.layers.Input(shape=(window, 1)),
        GRU(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        GRU(64),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss=custom_mse)
    return model

def modelo_profundo(window):
    model = Sequential([
        tf.keras.layers.Input(shape=(window, 1)),
        GRU(256, return_sequences=True),
        BatchNormalization(),
        Dropout(0.4),
        GRU(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        GRU(64),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.0005), loss=custom_mse)
    return model
