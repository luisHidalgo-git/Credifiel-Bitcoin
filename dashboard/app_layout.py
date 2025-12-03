from dash import dcc, html
import dash_bootstrap_components as dbc
import pandas as pd

def crear_pestana_modelo(nombre_modelo):
    return [
        dbc.Row([
            dbc.Col(dcc.Graph(id=f'{nombre_modelo.lower()}-predicciones'), md=6),
            dbc.Col(dcc.Graph(id=f'{nombre_modelo.lower()}-residuos'), md=6)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id=f'{nombre_modelo.lower()}-entrenamiento'), md=6),
            dbc.Col(dcc.Graph(id=f'{nombre_modelo.lower()}-distribucion-errores'), md=6)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id=f'{nombre_modelo.lower()}-metricas-temporales'), width=12)
        ])
    ]

def crear_layout(app, df, df_pred, metricas, ventana, price_col):
    return dbc.Container([
        html.H1("Comparación de Modelos Predictivos Bitcoin", className="text-center my-4"),

        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("GRU Simple", className="text-center"),
                dbc.CardBody([
                    html.H4(f"RMSE: {metricas['Simple']['RMSE']:.2f}", style={"color": "#FF5733"}, className="text-center text-primary"),
                    html.P(f"MAE: {metricas['Simple']['MAE']:.2f}", style={"color": "#66befb"}, className="text-center"),
                    html.P("Mejor modelo" if min(metricas.values(), key=lambda x: x['RMSE']) == metricas['Simple'] else "",
                           className="text-center text-success" if min(metricas.values(), key=lambda x: x['RMSE']) == metricas['Simple'] else "text-center")
                ]),
                dbc.CardFooter(f"Ventana: {ventana} días", className="text-center")
            ], color="primary" if min(metricas.values(), key=lambda x: x['RMSE']) == metricas['Simple'] else None), md=4),

            dbc.Col(dbc.Card([
                dbc.CardHeader("GRU Avanzado", className="text-center"),
                dbc.CardBody([
                    html.H4(f"RMSE: {metricas['Avanzado']['RMSE']:.2f}", style={"color": "#FF5733"}, className="text-center text-primary"),
                    html.P(f"MAE: {metricas['Avanzado']['MAE']:.2f}", style={"color": "#66befb"}, className="text-center"),
                    html.P("Mejor modelo" if min(metricas.values(), key=lambda x: x['RMSE']) == metricas['Avanzado'] else "",
                           className="text-center text-success" if min(metricas.values(), key=lambda x: x['RMSE']) == metricas['Avanzado'] else "text-center")
                ]),
                dbc.CardFooter(f"Ventana: {ventana} días", className="text-center")
            ], color="primary" if min(metricas.values(), key=lambda x: x['RMSE']) == metricas['Avanzado'] else None), md=4),

            dbc.Col(dbc.Card([
                dbc.CardHeader("GRU Profundo", className="text-center"),
                dbc.CardBody([
                    html.H4(f"RMSE: {metricas['Profundo']['RMSE']:.2f}", style={"color": "#FF5733"}, className="text-center text-primary"),
                    html.P(f"MAE: {metricas['Profundo']['MAE']:.2f}", style={"color": "#66befb"}, className="text-center"),
                    html.P("Mejor modelo" if min(metricas.values(), key=lambda x: x['RMSE']) == metricas['Profundo'] else "",
                           className="text-center text-success" if min(metricas.values(), key=lambda x: x['RMSE']) == metricas['Profundo'] else "text-center")
                ]),
                dbc.CardFooter(f"Ventana: {ventana} días", className="text-center")
            ], color="primary" if min(metricas.values(), key=lambda x: x['RMSE']) == metricas['Profundo'] else None), md=4),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                html.Label("Seleccione el rango de fechas:", className="mb-2"),
                dcc.DatePickerRange(
                    id='date-range',
                    min_date_allowed=df_pred.index.min(),
                    max_date_allowed=df_pred.index.max(),
                    start_date=df_pred.index.max() - pd.Timedelta(days=180),
                    end_date=df_pred.index.max(),
                    className='mb-4',
                    display_format='YYYY-MM-DD'
                )
            ], width=12)
        ], className="mb-3"),

        dcc.Tabs([
            dcc.Tab(label='Comparación General', children=[
                dbc.Row(dbc.Col(dcc.Graph(id='comparacion-modelos'), width=12)),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='errores-comparativos'), md=6),
                    dbc.Col(dcc.Graph(id='metricas-comparativas'), md=6)
                ]),
                dbc.Row(dbc.Col(dcc.Graph(id='residuos-modelos'), width=12))
            ]),

            dcc.Tab(label='Modelo Simple', children=crear_pestana_modelo('Simple')),
            dcc.Tab(label='Modelo Avanzado', children=crear_pestana_modelo('Avanzado')),
            dcc.Tab(label='Modelo Profundo', children=crear_pestana_modelo('Profundo'))
        ])
    ], fluid=True)
