import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import RegressorMixin

class DataPreprocessor:
    def __init__(self):
        self.imputer = SimpleImputer(strategy='mean') # Заполнение пропусков средним значением
        self.scaler = StandardScaler() # Масштабирование: среднее = 0, стандартное отклонение = 1

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df_imputed = self.imputer.fit_transform(df) # Заполняет пропуски
        df_scaled = self.scaler.fit_transform(df_imputed) # Масштабирует признаки
        return pd.DataFrame(df_scaled, columns=df.columns)