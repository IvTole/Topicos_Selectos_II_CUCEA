# Este script contiene los modelos (arquitecturas) de la red neuronal construida

import torch
from torch import nn
from  src.config import SEED

# Definimos semilla aleatoria en pytorch
torch.manual_seed(SEED)

class MNISTModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), # la entrada de la red neuronal va a ser en forma de vector de features, como se tenía en notas pasadas
            nn.Linear(in_features=input_shape, out_features=hidden_units), # in_features = numero de features en los datos (784 píxeles)
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )
    
    def forward(self, x):
        return self.layer_stack(x)

class MNISTModelV1(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(), # función de activación
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU() # función de activación
        )
    
    def forward(self, x: torch.Tensor):
        return self.layer_stack(x)

class MNISTModelV2(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()

        # Primera capa de convolucion
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, # tamaño del kernel/filtro (normalmente de 3)
                      stride=1, # default
                      padding=1),# opciones = "valid" o "same" 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, # Max Pooling
                         stride=2) # por default el valor del stride es igual al tamaño del kernel
        )

        # Segunda capa de convolucion
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), # Hasta este paso es cuando se aplana el tensor
            # Cada capa de la red comprime y cambia el shape de los datos de entrada
            nn.Linear(in_features=hidden_units*7*7, 
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        # print(x.shape)
        x = self.block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x