#!/Users/vanotole/Projects/Repos/Topicos_Selectos_II_CUCEA/.venv/bin/python

# Este es el script principal

# Librerías
import torch
from datetime import datetime
from torchsummary import summary

from src.get_data import get_data
from src.data_setup import create_dataloaders
from src.model_builder import MNISTModelV0, MNISTModelV1, MNISTModelV2
from src.engine import train
from src.config import LEARNING_RATE, EPOCHS

# Función principal
def main():
    print(datetime.now())

    # get data (training / test)
    train_data, test_data = get_data()

    classes = train_data.classes

    #print(train_data.classes)

    # DataLoader
    train_data_loader, test_data_loader = create_dataloaders(train_data=train_data,
                                                             test_data=test_data)

    # build model
    #model = MNISTModelV0(input_shape=784,
    #                     hidden_units=10,
    #                     output_shape=len(classes))
    model = MNISTModelV2(input_shape=1, 
                         hidden_units=10, 
                         output_shape=len(classes))
    print(summary(model, input_size=(1,28,28)))

    # Set up

    # Función de pérdida
    loss_fn = torch.nn.CrossEntropyLoss()

    # Stochastic Gradient Descend
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # training
    train(model=model,
          train_dataloader=train_data_loader,
          test_dataloader=test_data_loader,
          loss_fn=loss_fn,
          optimizer=optimizer,
          device="cpu",
          epochs = EPOCHS)

if __name__ == "__main__":
    main()