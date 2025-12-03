# Comparador de Modelos Predictivos Bitcoin

Sistema modularizado para entrenar y comparar tres arquitecturas de redes neuronales GRU para la predicción de precios de Bitcoin. Incluye un dashboard interactivo construido con Dash para visualizar y analizar el desempeño de los modelos.

## Características

-   **Tres modelos GRU diferentes**:

    -   **Simple**: Arquitectura básica con 1 capa GRU
    -   **Avanzado**: Modelo mejorado con 2 capas GRU, normalización y dropout
    -   **Profundo**: Arquitectura profunda con 3 capas GRU, regularización completa

-   **Dashboard interactivo** con:

    -   Comparación en tiempo real de predicciones vs valores reales
    -   Selector de rango de fechas
    -   Métricas RMSE y MAE por modelo
    -   Análisis de residuos y distribución de errores
    -   Curvas de aprendizaje durante el entrenamiento
    -   4 pestañas temáticas (Comparación General + 1 por modelo)

-   **Arquitectura modularizada**:
    -   Separación clara de responsabilidades
    -   Carga de datos
    -   Preprocesamiento
    -   Definición de modelos
    -   Entrenamiento
    -   Persistencia
    -   Dashboard interactivo

## Requisitos

```
tensorflow>=2.10.0
keras>=2.10.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
plotly>=5.0.0
dash>=2.0.0
dash-bootstrap-components>=1.0.0
joblib>=1.0.0
```

## Instalación

1. Clonar o descargar el proyecto

2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

3. Asegurarse de tener el archivo CSV de datos:

```
corr_bitcoin_diario_clausura.csv
```

El CSV debe contener una columna de precios con uno de estos nombres:

-   `clausura`
-   `close`
-   `Close`
-   `price`

## Estructura del Proyecto

```
project/
├── main.py                      # Punto de entrada principal
├── config/
│   ├── custom_loss.py          # Función de pérdida personalizada
│   └── settings.py             # Configuración global
├── data/
│   ├── load_data.py            # Carga de archivos CSV
│   └── preprocess.py           # Normalización y creación de ventanas
├── models/
│   ├── architectures.py        # Definición de arquitecturas GRU
│   └── train_models.py         # Lógica de entrenamiento
├── dashboard/
│   ├── build_dashboard.py      # Configuración principal del dashboard
│   ├── app_layout.py           # Estructura HTML/componentes
│   └── callbacks.py            # Lógica interactiva de gráficos
├── utils/
│   └── persistence.py          # Guardar/cargar datos
└── README.md
```

## Cómo Ejecutar

### 1. Entrenar Modelos y Ejecutar Dashboard

```bash
python main.py
```

Este comando:

-   Carga el archivo CSV
-   Normaliza los datos (0-1)
-   Crea ventanas temporales (14 días)
-   Entrena los 3 modelos con early stopping y reducción de learning rate
-   Guarda los modelos en archivos `.h5`
-   Persiste datos en `datos_dashboard.save`
-   Inicia el dashboard en `http://localhost:8050`

### 2. Acceder al Dashboard

Una vez ejecutado `python main.py`, abre tu navegador en:

```
http://localhost:8050
```

## Flujo de Ejecución

### Fase 1: Carga de Datos

```
CSV → Detección de columna precio → Extracción de serie de precios
```

### Fase 2: Preprocesamiento

```
Datos originales → Normalización MinMax (0-1) → Creación de ventanas (14 días)
```

### Fase 3: Entrenamiento

```
70% entrenamiento | 30% validación → Entrenamiento 100 épocas → Early Stopping
```

Parámetros de entrenamiento:

-   **Épocas**: 100 (con early stopping si no mejora)
-   **Batch size**: 32
-   **Validación**: 30% de los datos
-   **Optimizador**: Adam
-   **Loss**: MSE personalizado
-   **Callbacks**:
    -   EarlyStopping: Paciencia de 15 épocas
    -   ReduceLROnPlateau: Reduce learning rate cada 5 épocas sin mejora

### Fase 4: Evaluación

```
Predicciones sobre datos de test → Cálculo RMSE y MAE → Almacenamiento de métricas
```

### Fase 5: Dashboard

```
Carga modelos entrenados → Genera predicciones → Muestra visualizaciones interactivas
```

## Uso del Dashboard

### Pestaña: Comparación General

-   **Gráfico principal**: Superpone predicciones de los 3 modelos con valores reales
-   **Distribución de errores**: Box plots mostrando rango y outliers por modelo
-   **Métricas comparativas**: Barras con RMSE y MAE
-   **Análisis de residuos**: Scatter plot de errores vs predicciones

### Pestaña: Modelo Simple/Avanzado/Profundo

Cada modelo tiene 5 gráficos:

1. **Predicciones vs Real**: Línea temporal de predicciones superpuesta con valores reales
2. **Residuos**: Errores a lo largo del tiempo con línea de media
3. **Curva de aprendizaje**: Pérdida en entrenamiento vs validación
4. **Distribución de errores**: Histograma de errores
5. **Métricas temporales**: MAE con ventana móvil de 7 días

### Selector de Fechas

Filtra todos los gráficos por rango de fechas seleccionado. Por defecto muestra últimos 180 días.

## Parámetros Configurables

Para modificar parámetros, edita `main.py`:

```python
VENTANA = 14              # Tamaño de ventana temporal (días)
TRAIN_SIZE = 0.7          # Proporción de datos para entrenamiento
epochs = 100              # Número máximo de épocas
batch_size = 32           # Tamaño de lote
learning_rate = 0.001     # Tasa de aprendizaje (modelos avanzado/profundo)
```

## Interpretación de Métricas

### RMSE (Root Mean Squared Error)

-   Penaliza más los errores grandes
-   Mejor para penalizar predicciones muy alejadas
-   Unidades: USD
-   **Rango típico**: 100-500 para Bitcoin

### MAE (Mean Absolute Error)

-   Error promedio absoluto
-   Más robusto frente a outliers
-   Unidades: USD
-   **Rango típico**: 50-200 para Bitcoin

**Mejor modelo**: Aquel con RMSE y MAE menores

## Archivos Generados

Después de ejecutar, se crean:

```
modelo_Simple.h5          # Modelo Simple entrenado
modelo_Avanzado.h5        # Modelo Avanzado entrenado
modelo_Profundo.h5        # Modelo Profundo entrenado
datos_dashboard.save      # Datos persistidos (pickle con joblib)
```

Para limpiar y reentrenar:

```bash
rm modelo_*.h5 datos_dashboard.save
python main.py
```

## Solución de Problemas

### Error: "No se encontró columna de precios"

**Causa**: El CSV no tiene una columna con nombre `clausura`, `close`, `Close` o `price`

**Solución**:

1. Verifica los nombres exactos de columnas en tu CSV
2. Modifica la línea en `main.py`:

```python
price_col = next((col for col in ['tu_columna'] if col in df.columns), None)
```

### Error: "Índice datetime inválido"

**Causa**: El CSV no tiene fechas o están en formato no reconocido

**Solución**:

1. Asegúrate de que el CSV tenga una columna de fechas
2. O que el índice sea reconocible como fechas

### Dashboard no carga

**Causa**: Puerto 8050 ya está en uso

**Solución**:

```python
# En dashboard/build_dashboard.py, cambiar:
app.run(debug=True, port=8051)  # Usar otro puerto
```

### Memoria insuficiente

**Causa**: Dataset muy grande o modelo profundo

**Solución**:

1. Reducir batch_size de 32 a 16
2. Usar ventana más pequeña (10 en lugar de 14)
3. Usar modelo Simple en lugar de Profundo

## Ejemplo de Output

```
Archivo cargado. Columnas disponibles: ['fecha', 'clausura', 'volumen']
Columna de precio detectada: clausura
⚙️ Entrenando modelo Simple...
✔ Simple - RMSE: 245.32
⚙️ Entrenando modelo Avanzado...
✔ Avanzado - RMSE: 198.45
⚙️ Entrenando modelo Profundo...
✔ Profundo - RMSE: 156.78
Entrenamiento completado. Iniciando dashboard...
Dash is running on http://127.0.0.1:8050
```

## Notas de Rendimiento

-   **Tiempo de entrenamiento**: ~5-10 minutos en CPU, <2 minutos en GPU
-   **Tiempo de carga del dashboard**: <5 segundos
-   **Tiempo de actualización de gráficos**: <2 segundos por cambio de rango de fechas

## Mejoras Futuras

-   [ ] Agregar más modelos (LSTM, Transformer)
-   [ ] Exportar predicciones a CSV
-   [ ] Predicciones futuras (forecasting)
-   [ ] Backtesting de estrategias
-   [ ] Integración con APIs en tiempo real
-   [ ] Configuración por archivo YAML
-   [ ] Tests unitarios

## Licencia

Proyecto educativo sin restricciones de uso.

## Contacto

Para reportar bugs o sugerencias, contacta al desarrollador.
