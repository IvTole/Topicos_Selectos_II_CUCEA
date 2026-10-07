import torch
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

# Setup datos de entrenamiento
train_data = datasets.MNIST(
    root="./", # dónde bajar los datos
    train=True, # obtener datos de entrenamiento
    download=True, # download si no se encuentra
    transform=ToTensor(), # images vienen en formato PIL, pasar a tensores de Torch
    target_transform=None
)

# Setup datos de prueba
test_data = datasets.MNIST(
    root="./",
    train=False, # obtener datos de validación
    download=True,
    transform=ToTensor()
)