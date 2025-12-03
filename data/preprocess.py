import numpy as np
from sklearn.preprocessing import MinMaxScaler

def seleccionar_columna_precio(df):
    columnas = ['clausura', 'close', 'Close', 'price']
    for col in columnas:
        if col in df.columns:
            return df[col].values.reshape(-1, 1), col
    raise ValueError("No se encontró columna de precios válida.")

def crear_ventanas_temporales(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size, 0])
        y.append(data[i+window_size, 0])
    return np.array(X).reshape(-1, window_size, 1), np.array(y).reshape(-1, 1)

def normalizar_datos(datos):
    scaler = MinMaxScaler()
    datos_norm = scaler.fit_transform(datos)
    return datos_norm, scaler
