# Este script contiene los modelos (arquitecturas) de la red neuronal construida

import torch
from torch import nn
from  src.config import SEED

# Definimos semilla aleatoria en pytorch
torch.manual_seed(SEED)

class Model_Classification_V1(nn.Module):

    # constructor
    def __init__(self, input_shape:int, hidden_units:int, output_features:int):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=input_shape, out_features=hidden_units)
        self.layer_2 = nn.Linear(in_features=hidden_units, out_features=hidden_units)
        self.layer_3 = nn.Linear(in_features=hidden_units, out_features=output_features)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    # Forward Propagation
    def forward(self, x):
        return self.softmax(self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x))))))