# Este script tiene la responsabilidad de carga de datos (como tensores de pytorch)

import torch
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

from src.config import DATA_PATH

def get_data():
    """
    """

    # Setup datos de entrenamiento
    train_data = datasets.MNIST(
        root=DATA_PATH, # dónde bajar los datos
        train=True, # obtener datos de entrenamiento
        download=True, # download si no se encuentra
        transform=ToTensor(), # images vienen en formato PIL, pasar a tensores de Torch
        target_transform=None
    )

    # Setup datos de prueba
    test_data = datasets.MNIST(
        root=DATA_PATH,
        train=False, # obtener datos de validación
        download=True,
        transform=ToTensor()
    )

    return train_data, test_data