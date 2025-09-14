from sklearn.datasets import load_linnerud
from sklearn.utils import Bunch
import pandas as pd

class DataLoader:
    def load(self):
        """Загрузка данных из Linnerud датасет.
        
        :return: Физиологические признаки и спортивные результаты.
        """
        data: Bunch = load_linnerud() # Загружает датасет Linnerud как Bunch-объект
        x = pd.DataFrame(data.target, columns=data.target_names) # Физиологические признаки (X)
        y = pd.DataFrame(data.data, columns=data.feature_names) # Спортивные результаты (Y)
        return x, y