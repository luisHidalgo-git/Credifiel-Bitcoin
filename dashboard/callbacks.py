from dash import Input, Output
import plotly.graph_objs as go

def registrar_callbacks(app, df, df_pred, modelos, metricas, historias, price_col):

    @app.callback(
        [Output('comparacion-modelos', 'figure'),
         Output('errores-comparativos', 'figure'),
         Output('metricas-comparativas', 'figure'),
         Output('residuos-modelos', 'figure')] +
        [Output(f'{nombre.lower()}-predicciones', 'figure') for nombre in modelos.keys()] +
        [Output(f'{nombre.lower()}-residuos', 'figure') for nombre in modelos.keys()] +
        [Output(f'{nombre.lower()}-entrenamiento', 'figure') for nombre in modelos.keys()] +
        [Output(f'{nombre.lower()}-distribucion-errores', 'figure') for nombre in modelos.keys()] +
        [Output(f'{nombre.lower()}-metricas-temporales', 'figure') for nombre in modelos.keys()],
        [Input('date-range', 'start_date'),
         Input('date-range', 'end_date')]
    )
    def actualizar_todo(start_date, end_date):
        filtered = df_pred[start_date:end_date]

        colores = {'Simple': '#1f77b4', 'Avanzado': '#ff7f0e', 'Profundo': '#2ca02c'}

        # 1. Gráfico comparativo principal
        fig_comparacion = go.Figure()
        fig_comparacion.add_trace(go.Scatter(
            x=filtered.index,
            y=filtered[price_col],
            name='Valor Real',
            line=dict(color='black', width=3),
            mode='lines+markers'
        ))

        for nombre in modelos.keys():
            fig_comparacion.add_trace(go.Scatter(
                x=filtered.index,
                y=filtered[f'pred_{nombre.lower()}'],
                name=f"{nombre} (RMSE: {metricas[nombre]['RMSE']:.2f})",
                line=dict(color=colores[nombre], width=1.5),
                mode='lines+markers'
            ))

        fig_comparacion.update_layout(
            title="Comparación de Predicciones vs Real",
            xaxis_title="Fecha",
            yaxis_title="Precio (USD)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # 2. Gráfico de errores comparativos
        fig_errores = go.Figure()
        for nombre in modelos.keys():
            error = abs(filtered[price_col] - filtered[f'pred_{nombre.lower()}'])
            fig_errores.add_trace(go.Box(
                y=error,
                name=nombre,
                marker_color=colores[nombre],
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))

        fig_errores.update_layout(
            title="Distribución de Errores Absolutos por Modelo",
            yaxis_title="Error Absoluto (USD)",
            showlegend=False
        )

        # 3. Gráfico de métricas comparativas
        fig_metricas = go.Figure()
        metricas_plot = ['RMSE', 'MAE']
        for metrica in metricas_plot:
            fig_metricas.add_trace(go.Bar(
                x=list(modelos.keys()),
                y=[metricas[nombre][metrica] for nombre in modelos.keys()],
                name=metrica,
                text=[f"{metricas[nombre][metrica]:.2f}" for nombre in modelos.keys()],
                textposition='auto'
            ))

        fig_metricas.update_layout(
            title="Métricas Comparativas",
            yaxis_title="Valor",
            barmode='group'
        )

        # 4. Gráfico de residuos
        fig_residuos = go.Figure()
        for nombre in modelos.keys():
            residuos = filtered[price_col] - filtered[f'pred_{nombre.lower()}']
            fig_residuos.add_trace(go.Scatter(
                x=filtered[f'pred_{nombre.lower()}'],
                y=residuos,
                name=nombre,
                mode='markers',
                marker=dict(color=colores[nombre])
            ))

        fig_residuos.update_layout(
            title="Análisis de Residuos",
            xaxis_title="Predicciones",
            yaxis_title="Residuos (Real - Predicción)",
            showlegend=True
        )
        fig_residuos.add_hline(y=0, line_dash="dash", line_color="grey")

        # Gráficos individuales para cada modelo
        figs_individuales = []
        for nombre in modelos.keys():
            # Predicciones vs Real
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=filtered.index,
                y=filtered[price_col],
                name='Real',
                line=dict(color='black'),
                mode='lines+markers'
            ))
            fig_pred.add_trace(go.Scatter(
                x=filtered.index,
                y=filtered[f'pred_{nombre.lower()}'],
                name='Predicción',
                line=dict(color=colores[nombre]),
                mode='lines+markers'
            ))
            fig_pred.update_layout(
                title=f"Predicciones vs Real - {nombre}",
                xaxis_title="Fecha",
                yaxis_title="Precio (USD)",
                hovermode="x unified"
            )
            figs_individuales.append(fig_pred)

            # Gráfico de residuos individual
            residuos = filtered[price_col] - filtered[f'pred_{nombre.lower()}']
            fig_resid = go.Figure()
            fig_resid.add_trace(go.Scatter(
                x=filtered.index,
                y=residuos,
                name='Residuos',
                line=dict(color=colores[nombre]),
                mode='lines+markers'
            ))
            fig_resid.update_layout(
                title=f"Residuos - {nombre}",
                xaxis_title="Fecha",
                yaxis_title="Error (Real - Predicción)",
                showlegend=False
            )
            fig_resid.add_hline(y=0, line_dash="dash", line_color="grey")
            fig_resid.add_hline(y=residuos.mean(), line_dash="dot", line_color="red")
            figs_individuales.append(fig_resid)

            # Curva de aprendizaje
            fig_train = go.Figure()
            fig_train.add_trace(go.Scatter(
                y=historias[nombre]['loss'],
                name='Entrenamiento',
                line=dict(color=colores[nombre])
            ))
            fig_train.add_trace(go.Scatter(
                y=historias[nombre]['val_loss'],
                name='Validación',
                line=dict(color=colores[nombre], dash='dot')
            ))
            fig_train.update_layout(
                title=f"Curva de Aprendizaje - {nombre}",
                xaxis_title="Época",
                yaxis_title="MSE",
                legend=dict(orientation="h")
            )
            figs_individuales.append(fig_train)

            # Distribución de errores
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=residuos,
                name='Errores',
                marker_color=colores[nombre],
                opacity=0.75
            ))
            fig_dist.update_layout(
                title=f"Distribución de Errores - {nombre}",
                xaxis_title="Error",
                yaxis_title="Frecuencia",
                bargap=0.1
            )
            figs_individuales.append(fig_dist)

            # Métricas temporales
            error_rolling = abs(residuos).rolling(window=7).mean()
            fig_metrics = go.Figure()
            fig_metrics.add_trace(go.Scatter(
                x=filtered.index,
                y=error_rolling,
                name='MAE (7 días)',
                line=dict(color=colores[nombre])
            ))
            fig_metrics.update_layout(
                title=f"Error Absoluto Medio (ventana 7 días) - {nombre}",
                xaxis_title="Fecha",
                yaxis_title="MAE",
                showlegend=False
            )
            figs_individuales.append(fig_metrics)

        return [fig_comparacion, fig_errores, fig_metricas, fig_residuos] + figs_individuales
