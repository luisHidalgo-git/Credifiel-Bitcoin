from dash import Input, Output
import plotly.graph_objs as go
from config.settings import price_col

def registrar_callbacks(app, df, df_pred, modelos, metricas):

    @app.callback(
        Output("comparacion-modelos", "figure"),
        Input("comparacion-modelos", "id")
    )
    def actualizar(_):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_pred.index,
            y=df_pred[price_col],
            name="Real",
            mode="lines"
        ))
        for nombre in modelos:
            fig.add_trace(go.Scatter(
                x=df_pred.index,
                y=df_pred[f'pred_{nombre.lower()}'],
                name=nombre
            ))
        return fig
