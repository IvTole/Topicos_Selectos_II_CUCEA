#!/Users/vanotole/Projects/Repos/Topicos_Selectos_II_CUCEA/.venv/bin/python

# Este es el script principal

# Librerías
import torch
from datetime import datetime
from torchsummary import summary

from src.get_data import get_data
from src.model_builder import Model_Classification_V1
from src.engine import train
from src.config import LEARNING_RATE, EPOCHS

# Función principal
def main():
    print(datetime.now())

    # get data (training / test)
    X_train, y_train, X_test, y_test = get_data()
    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    # build model
    model = Model_Classification_V1(input_shape=64,
                                    hidden_units=10,
                                    output_features=10)
    print(summary(model, input_size=(64,)))

    # Set up

    # Función de pérdida
    loss_fn = torch.nn.CrossEntropyLoss()

    # Stochastic Gradient Descend
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # training
    train(model=model,
          X_train=X_train,
          y_train=y_train,
          X_test = X_test,
          y_test = y_test,
          loss_fn=loss_fn,
          optimizer=optimizer,
          device="cpu",
          epochs = EPOCHS)

if __name__ == "__main__":
    main()