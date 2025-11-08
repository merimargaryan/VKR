import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def save_models(models_dict, filepath):
    """Сохранение моделей и preprocessing объектов"""
    for name, model in models_dict.items():
        joblib.dump(model, f"{filepath}/{name.replace(' ', '_').lower()}.pkl")
    
    print("Модели успешно сохранены")

def load_models(filepath):
    """Загрузка моделей"""
    models = {}
    # Загрузка логики здесь
    return models


def preprocess_data(df):
    """Функция предобработки как в ноутбуке"""
    # Реализация предобработки
    return processed_df