from dash import Dash
import dash_bootstrap_components as dbc
from .app_layout import crear_layout
from .callbacks import registrar_callbacks

def iniciar_dashboard(df, df_pred, modelos, metricas, historias, ventana, price_col):
    app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
    app.title = "Comparador de Modelos Bitcoin"

    app.layout = crear_layout(app, df, df_pred, metricas, ventana, price_col)
    registrar_callbacks(app, df, df_pred, modelos, metricas, historias, price_col)

    app.run(debug=True, port=8050)

