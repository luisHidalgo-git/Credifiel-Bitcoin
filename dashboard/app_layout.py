from dash import dcc, html
import dash_bootstrap_components as dbc
from config.settings import price_col

def crear_layout(app, df, metricas, ventana):
    return dbc.Container([
        html.H1("Comparador de Modelos Bitcoin", className="text-center my-4"),

        dcc.Tabs([
            dcc.Tab(label="Comparación General", children=[
                dcc.Graph(id="comparacion-modelos")
            ]),
            dcc.Tab(label="Modelo Simple", children=[
                dcc.Graph(id="simple-grafico")
            ])
        ])
    ], fluid=True)
