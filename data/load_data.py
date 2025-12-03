# data/load_data.py
import pandas as pd

def cargar_csv(ruta_csv, fecha_cols=None):
    """
    Carga un CSV intentando detectar y normalizar la(s) columna(s) de fecha.
    - ruta_csv: ruta al CSV
    - fecha_cols: lista opcional de nombres de columnas que podrían ser la fecha (p.ej. ['fecha','date'])
    Devuelve: df (sin índice modificado todavía) y nombre_columna_fecha (o None)
    """
    # Primer intento: cargar sin parseo para poder inspeccionar columnas
    df = pd.read_csv(ruta_csv)
    print("✓ Archivo cargado. Columnas disponibles:", df.columns.tolist())

    # Si nos dieron columnas candidatas, verifícalas
    candidata = None
    if fecha_cols:
        for c in fecha_cols:
            if c in df.columns:
                candidata = c
                break

    # Si no hay candidata especificada, intenta encontrar columnas típicas
    if candidata is None:
        for c in ['fecha', 'date', 'timestamp', 'Fecha', 'DATE']:
            if c in df.columns:
                candidata = c
                break

    # Si encontramos una columna candidata, rehacemos la carga con parse_dates para mayor robustez
    if candidata is not None:
        try:
            df = pd.read_csv(ruta_csv, parse_dates=[candidata])
            print(f"✓ Columna de fecha detectada: '{candidata}' (convertida a datetime).")
            return df, candidata
        except Exception as e:
            print("⚠️ No se pudo parsear la columna de fecha automáticamente:", e)
            # devolvemos df sin parseo explícito, pero indicamos la candidata
            return df, candidata

    # Si no encontramos ninguna columna de fecha
    return df, None
