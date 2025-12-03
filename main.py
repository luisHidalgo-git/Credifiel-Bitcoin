# main.py (fragmento: reemplazar la parte de carga y preparación)
from data.load_data import cargar_csv
from data.preprocess import seleccionar_columna_precio, crear_ventanas_temporales, normalizar_datos
from models.train_models import entrenar_todos
from utils.persistence import cargar_datos_dashboard
from dashboard.build_dashboard import iniciar_dashboard
from keras.models import load_model
import pandas as pd
import numpy as np

print("Iniciando entrenamiento...")

# 1) Cargar CSV tratando de detectar la columna de fecha
df, fecha_col = cargar_csv("corr_bitcoin_diario_clausura.csv", fecha_cols=['fecha','date','fecha_hora'])

# 2) Si se detectó columna de fecha, asegurar que sea datetime y ponerla como índice
if fecha_col is not None:
    # Si la columna ya está en datetime por parse_dates, lo dejará como datetime
    try:
        df[fecha_col] = pd.to_datetime(df[fecha_col], errors='coerce')
    except Exception as e:
        print("⚠️ Error al convertir columna de fecha con to_datetime:", e)

    # Mostrar si hay valores NaT (errores de parseo)
    n_nans = df[fecha_col].isna().sum()
    if n_nans > 0:
        print(f"⚠️ Atención: {n_nans} filas tienen fecha inválida (NaT). Revisa el CSV.")
    # Establecer índice
    df.set_index(fecha_col, inplace=True)
    print("✓ Índice de fechas establecido. Rango:", df.index.min(), "->", df.index.max())
else:
    # Si no hay columna fecha, intentamos transformar el índice actual a datetime (como en monolítico)
    try:
        df.index = pd.to_datetime(df.index)
        print("✓ Índice convertido con éxito a datetime a partir del índice existente.")
    except Exception as e:
        print("⚠️ No se detectó columna fecha y no se pudo convertir el índice a datetime:", e)

# Comprobación: asegurarnos de que la columna de precio existe
datos_col, price_col = seleccionar_columna_precio(df.reset_index())  # la función espera df con columna, no índice
# NOTA: seleccionar_columna_precio devuelve los valores y el nombre encontrado; la usamos solo para validar
if not price_col:
    raise ValueError("No se encontró columna de precio. Asegúrate que exista 'clausura' o 'close' en el CSV.")
print("✓ Columna de precio detectada:", price_col)

# 3) Normalización y ventanas (igual que antes)
# Extraer la serie de precio directamente desde el df con el nombre 'price_col'
datos = df[price_col].values.reshape(-1, 1)
datos_norm, scaler = normalizar_datos(datos)

VENTANA = 14
X, y = crear_ventanas_temporales(datos_norm, VENTANA)

split_idx = int(len(X) * 0.7)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 4) Entrenar modelos (igual que antes)
metricas = entrenar_todos(X_train, y_train, X_test, y_test, scaler, df, split_idx, VENTANA)

print("Entrenamiento completado. Cargando datos para dashboard...")

# 5) Cargar datos guardados (los generó train_models)
datos_dash = cargar_datos_dashboard()

# 6) Asegurar que df en datos_dash tiene índice datetime (por si fue guardado sin cambios)
df_dash = datos_dash['df']
# Si en el dump df fue guardado con índice como columna, intentar lo mismo:
if not isinstance(df_dash.index.dtype, (pd.DatetimeTZDtype, pd.core.dtypes.dtypes.DatetimeTZDtype)) and not pd.api.types.is_datetime64_any_dtype(df_dash.index):
    # intentar convertir la columna 'fecha' si existe
    if 'fecha' in df_dash.columns:
        df_dash['fecha'] = pd.to_datetime(df_dash['fecha'], errors='coerce')
        df_dash.set_index('fecha', inplace=True)
    else:
        try:
            df_dash.index = pd.to_datetime(df_dash.index)
        except Exception:
            print("⚠️ df_dash no tiene índice datetime; gráficas pueden fallar.")

# 7) Reconstruir df_pred exactamente como antes (alineación por índice)
# df_pred debe comenzar en VENTANA + split_idx (como en el original)
df_pred = df_dash.iloc[VENTANA + split_idx:].copy()
# Asignar columna real de precios al df_pred desde df_dash (ya está indexado por fecha)
df_pred[price_col] = df_dash[price_col].iloc[VENTANA + split_idx:].values

# 8) Cargar modelos y añadir predicciones (igual)
modelos = {
    nombre: load_model(f"modelo_{nombre}.h5")
    for nombre in ['Simple', 'Avanzado', 'Profundo']
}

for nombre in modelos.keys():
    y_pred = modelos[nombre].predict(datos_dash['X_test'])
    df_pred[f'pred_{nombre.lower()}'] = datos_dash["scaler"].inverse_transform(y_pred).flatten()

# 9) Debug prints finales (útiles para verificar)
print("df_pred index head:", df_pred.index[:3])
print("df_pred index dtype:", df_pred.index.dtype)
print("df_pred columns:", df_pred.columns.tolist())

# 10) Iniciar dashboard
iniciar_dashboard(df_dash, df_pred, modelos, metricas, VENTANA)
