# Este script tiene la responsabilidad de carga de datos (como tensores de pytorch)

import torch
import pandas as pd
import os
from src.config import TRAIN_PATH, TEST_PATH

def get_data():
    """
    Descripcion:
    Carga de datos. Carga los datos en pandas y transforma a tensores de pytorch.

    Retorna:
    X_train - features de entrenamiento
    X_test - featuers de prueba
    y_train - target de entrenamiento
    y_test - target de prueba
    """

    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)

    # Dataframes de pandas (entrenamiento / prueba)
    X_train = df_train.drop(columns="target")
    y_train = df_train["target"]
    X_test = df_test.drop(columns="target")
    y_test = df_test["target"]

    # pasar a numpy
    X_train_array = X_train.to_numpy().copy()
    y_train_array = y_train.to_numpy().copy()
    X_test_array = X_test.to_numpy().copy()
    y_test_array = y_test.to_numpy().copy()

    # pasar a arreglos de pytorch
    X_train_torch = torch.from_numpy(X_train_array).type(torch.float32)
    y_train_torch = torch.from_numpy(y_train_array).type(torch.int64)
    X_test_torch = torch.from_numpy(X_test_array).type(torch.float32)
    y_test_torch = torch.from_numpy(y_test_array).type(torch.int64)

    #print(os.getcwd())

    return X_train_torch, y_train_torch, X_test_torch, y_test_torch