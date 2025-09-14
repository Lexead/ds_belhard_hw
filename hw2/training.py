from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from trainer import Trainer
from data_visualizer import DataVisualizer
import pandas as pd
import panel as pn
pn.extension('plotly')

# 1. Загрузка
x_raw, y_raw = DataLoader().load()
target_column = 'Chins' # Chins - Количество подтягиваний
y = y_raw[target_column]

# 2. Предобработка
x = DataPreprocessor().preprocess(x_raw)

# 3. Разделение, Обучение, Предсказание и Оценка учителем
trainer = Trainer()
x_train, x_test, y_train, y_test = trainer.split(x, y)
trainer.fit(x_train, y_train)
y_pred = trainer.predict(x_test)
metrics = trainer.evaluate(y_test, y_pred)

# 4. Визуализация
visualizer = DataVisualizer()
metrics = metrics.T.reset_index()
metrics.columns = ['Метрика', 'Значение']
tabs = pn.Tabs(
    ("Предсказания", pn.pane.Plotly(visualizer.plot_predictions(pd.Series(y_test, name="Реальные"), pd.Series(y_pred, name='Предсказанные')))),
    ("Распределение", pn.pane.Plotly(visualizer.plot_target_distribution(y))),
    ("Зависимость признака", pn.pane.Plotly(visualizer.plot_feature_vs_target(x, y, feature='Weight'))),
    ("Корреляции", pn.pane.Plotly(visualizer.plot_correlation_heatmap(x, y))),
    ("Оценка модели", visualizer.plot_metrics_table(metrics))
)

# pn.serve(tabs, show=True, title="Анализ подтягиваний", port=5006)
