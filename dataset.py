import numpy as np
import pandas as pd

class Dataset:
    def __init__(self,csv_path):
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path)
    def get_features(self):
        x = self.data["hours"].values
        y = self.data["score"].values
        x = np.array(x).reshape(-1, 1)
        y = np.array(y).reshape(-1, 1)
        return x, y

 