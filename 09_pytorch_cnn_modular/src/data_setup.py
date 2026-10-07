import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.config import BATCH_SIZE

def create_dataloaders(train_data,
                       test_data):
    train_dataloader = DataLoader(train_data, # dataset -> iterable
                                  batch_size=BATCH_SIZE, # cuántas muestras por batch 
                                  shuffle=True # shuffle para cada época
                                 )

    test_dataloader = DataLoader(test_data,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False
                                 )

    # Resumen
    print(f"Dataloaders: {train_dataloader, test_dataloader}") 
    print(f"Train dataloader (numero de batches): {len(train_dataloader)} batches of {BATCH_SIZE}")
    print(f"Test dataloader (numero de batches): {len(test_dataloader)} batches of {BATCH_SIZE}")

    return train_dataloader, test_dataloader