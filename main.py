# main.py
from data.preprocess import crear_ventanas_temporales, normalizar_datos
from models.train_models import entrenar_todos
from utils.persistence import cargar_datos_dashboard
from dashboard.build_dashboard import iniciar_dashboard
from keras.models import load_model
import pandas as pd
import numpy as np

print("Iniciando entrenamiento...")

# Cargar CSV
df = pd.read_csv('corr_bitcoin_diario_clausura.csv')
print("Archivo cargado. Columnas disponibles:", df.columns.tolist())

# Detectar columna de precio
price_col = next((col for col in ['clausura', 'close', 'Close', 'price'] if col in df.columns), None)
if not price_col:
    raise ValueError("No se encontró columna de precios en el CSV")
print("Columna de precio detectada:", price_col)

# Extraer datos de precio
datos = df[price_col].values.reshape(-1, 1)

# Normalización y ventanas
datos_norm, scaler = normalizar_datos(datos)

VENTANA = 14
X, y = crear_ventanas_temporales(datos_norm, VENTANA)

split_idx = int(len(X) * 0.7)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Entrenar modelos
metricas = entrenar_todos(X_train, y_train, X_test, y_test, scaler, df, split_idx, VENTANA, price_col)

print("Entrenamiento completado. Iniciando dashboard...")

# Cargar datos guardados
datos_guardados = cargar_datos_dashboard()
scaler = datos_guardados['scaler']
df_dash = datos_guardados['df']

# Configurar índice de fechas
df_dash.index = pd.to_datetime(df_dash.index)
if 'fecha' in df_dash.columns:
    df_dash['fecha'] = pd.to_datetime(df_dash['fecha'])
    df_dash.set_index('fecha', inplace=True)

split_idx = datos_guardados['split_idx']
VENTANA = datos_guardados['VENTANA']
metricas = datos_guardados['metricas']
historias = datos_guardados['historias']

# Preparar DataFrame para visualización
df_pred = df_dash.iloc[VENTANA + split_idx:].copy()
df_pred[price_col] = df_dash[price_col].iloc[VENTANA + split_idx:].values

# Cargar modelos y añadir predicciones
modelos = {
    nombre: load_model(f"modelo_{nombre}.h5")
    for nombre in ['Simple', 'Avanzado', 'Profundo']
}

for nombre in modelos.keys():
    y_pred = modelos[nombre].predict(datos_guardados['X_test'])
    df_pred[f'pred_{nombre.lower()}'] = scaler.inverse_transform(y_pred).flatten()

# Iniciar dashboard
iniciar_dashboard(df_dash, df_pred, modelos, metricas, historias, VENTANA, price_col)
