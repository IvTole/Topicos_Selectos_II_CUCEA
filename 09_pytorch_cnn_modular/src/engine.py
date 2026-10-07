import torch
from torchmetrics.classification import MulticlassAccuracy

def train_step(model: torch.nn.Module,
               dataloader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):

    # 1 - Modo de entrenamient
    model.train()

    # Inicializar contares de loss, accuracy
    train_loss, train_acc = 0, 0

    # Loop por batch

    for batch, (X,y) in enumerate(dataloader):

        # 2 - Poner los datos en el hardware
        X, y = X.to(device), y.to(device)

        # 3 - Forward Propagation
        y_pred = model(X)

        # 4 - Calcula perdida
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 5 - Zero grad optimizer
        optimizer.zero_grad()

        # 6 - BackPropagation
        loss.backward()

        # 7 - Actualizar pesos
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

  # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc

def test_step(model: torch.nn.Module,
               dataloader,
               loss_fn: torch.nn.Module,
               device: torch.device):

    # 1 - Modo de inferencia
    model.eval()

    # Acumuladores loss, accuracy
    test_loss, test_acc = 0, 0

    with torch.inference_mode():

        # loop por lote

        for batch, (X,y) in enumerate(dataloader):

            # 2 - Poner los datos en el hardware
            X, y = X.to(device), y.to(device)

            # 3 - Forward Propagation
            y_pred = model(X)
            #print(y_pred)

            # 4 - Calcula perdida
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = y_pred.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
          
        # Adjust metrics to get average loss and accuracy per batch 
        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)

    return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader,
          test_dataloader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          device: torch.device,
          epochs:int):

    # Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    for epoch in range(0,epochs):

        train_loss, train_acc = train_step(
            model = model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )


        test_loss, test_acc = test_step(
                    model = model,
                    dataloader=test_dataloader,
                    loss_fn=loss_fn,
                    device=device
                )
        
        if epoch % 1 == 0:

            print(
                    f"Epoch: {epoch+1} | "
                    f"train_loss: {train_loss:.4f} | "
                    f"train_acc: {train_acc:.4f} | "
                    f"test_loss: {test_loss:.4f} | "
                    f"test_acc: {test_acc:.4f}"
                    )

    return results
            

