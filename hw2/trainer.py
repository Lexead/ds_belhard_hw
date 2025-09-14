from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
import pandas as pd

class Trainer:
    def __init__(self):
        self.model = LinearRegression() # Инициализация модели линейной регрессии

    def split(self, X, y, test_size=0.3, random_state=42):
        return train_test_split(X, y, test_size=test_size, random_state=random_state) # Делит данные на обучающую и тестовую выборки (test_size=0.3 → 30% данных идут в тест, random_state=42 → фиксирует случайность для воспроизводимости)

    def fit(self, x, y):
        self.model.fit(x, y) # Обучение модели на признаках x и целевой переменной y

    def predict(self, x):
        return self.model.predict(x) # Предсказание на новых данных x

    def evaluate(self, y_true, y_pred):
        return pd.DataFrame({
            'R2': [r2_score(y_true, y_pred)], # Коэффициент детерминации: насколько хорошо модель объясняет данные
            'MSE': [mean_squared_error(y_true, y_pred)] # Среднеквадратичная ошибка: насколько сильно модель ошибается
        })

    def get_model(self):
        return self.model