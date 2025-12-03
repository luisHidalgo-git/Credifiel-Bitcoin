from dash import Dash
import dash_bootstrap_components as dbc
from .app_layout import crear_layout
from .callbacks import registrar_callbacks

def iniciar_dashboard(df, df_pred, modelos, metricas, ventana):
    app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
    app.title = "Bitcoin Dashboard"

    app.layout = crear_layout(app, df, metricas, ventana)
    registrar_callbacks(app, df, df_pred, modelos, metricas)

    app.run(debug=True)

