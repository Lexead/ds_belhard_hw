import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

class DataVisualizer:
    def plot_predictions(self, y_true: pd.Series, y_pred: pd.Series, title: str = "Сравнение фактических и предсказанных значений подтягиваний"):
        df = pd.DataFrame({'Реальные': y_true, 'Предсказанные': y_pred})
        fig = px.scatter(df, x='Реальные', y='Предсказанные', title=title, opacity=0.7)
        fig.add_shape(type='line',
                      x0=df['Реальные'].min(), y0=df['Реальные'].min(),
                      x1=df['Реальные'].max(), y1=df['Реальные'].max(),
                      line=dict(color='red', dash='dash'))
        return fig

    def plot_target_distribution(self, y: pd.Series, title: str = "Распределение подтягиваний"):
        """Гистограмма распределения целевой переменной (Chins).
        """
        fig = px.histogram(y, nbins=10, title=title,
                       labels={'value': 'Количество подтягиваний'},
                       color_discrete_sequence=['skyblue'])
        fig.update_layout(yaxis_title='Частота')
        return fig

    def plot_feature_vs_target(self, x: pd.DataFrame, y: pd.Series, feature: str):
        """Точечная диаграмма зависимости исходного признака от целевой переменной (Chins).
        """
        df = pd.DataFrame({f"Признак «{feature}»": x[feature], 'Количество подтягиваний': y})
        fig = px.scatter(df, x=f"Признак «{feature}»", y='Количество подтягиваний',
                        title=f"Зависимость признака «{feature}» от подтягиваний", opacity=0.7)
        return fig

    def plot_correlation_heatmap(self, x: pd.DataFrame, y: pd.Series, title: str = "Корреляции между физическими признаками и подтягиванием"):
        """Тепловая карта корреляций признаков и целевой переменной (Chins).
        """
        df = x.copy()
        df['Chins'] = y
        corr = df.corr().round(2)
        fig = ff.create_annotated_heatmap(z=corr.values,
                                        x=corr.columns.tolist(),
                                        y=corr.index.tolist(),
                                        colorscale='RdBu',
                                        showscale=True,
                                        reversescale=True,
                                        annotation_text=corr.values.astype(str))
        fig.update_layout(title_text=title)
        return fig
    
    def plot_metrics_table(self, results: pd.DataFrame, title: str = "Оценка модели"):
        fig = go.Figure(data=[go.Table(
        header=dict(values=list(results.columns),
                    fill_color='lightgrey',
                    align='left'),
        cells=dict(values=[results[col].tolist() for col in results.columns],
                   fill_color='white',
                   align='left'))
        ])
        fig.update_layout(title=title)
        return fig