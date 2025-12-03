import math
import joblib
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .architectures import modelo_simple, modelo_avanzado, modelo_profundo

def entrenar_todos(X_train, y_train, X_test, y_test, scaler, df, split_idx, ventana):

    modelos = {
        "Simple": modelo_simple(ventana),
        "Avanzado": modelo_avanzado(ventana),
        "Profundo": modelo_profundo(ventana)
    }

    early = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)

    metricas = {}
    historias = {}
    predicciones = {}

    for nombre, modelo in modelos.items():
        print(f"⚙️ Entrenando modelo {nombre}...")

        h = modelo.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            verbose=0,
            callbacks=[early, reduce]
        )

        modelo.save(f"modelo_{nombre}.h5")

        y_pred = modelo.predict(X_test)
        y_pred_inv = scaler.inverse_transform(y_pred)
        y_test_inv = scaler.inverse_transform(y_test)

        metricas[nombre] = {
            "RMSE": math.sqrt(mean_squared_error(y_test_inv, y_pred_inv)),
            "MAE": mean_absolute_error(y_test_inv, y_pred_inv)
        }

        historias[nombre] = h.history

        print(f"✔ {nombre} - RMSE: {metricas[nombre]['RMSE']:.2f}")

    joblib.dump({
        "scaler": scaler,
        "X_test": X_test,
        "y_test": y_test,
        "df": df,
        "split_idx": split_idx,
        "VENTANA": ventana,
        "metricas": metricas,
        "historias": historias
    }, "datos_dashboard.save")

    return metricas
