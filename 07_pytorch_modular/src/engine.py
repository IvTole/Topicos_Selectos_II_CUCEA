import torch

def train_step(model: torch.nn.Module,
               X,
               y,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):

    # 1 - Modo de entrenamient
    model.train()

    # 2 - Poner los datos en el hardware
    X, y = X.to(device), y.to(device)

    # 3 - Forward Propagation
    y_pred = model(X)

    # 4 - Calcula perdida
    loss_fn = loss_fn(y_pred, y)

    # 5 - Zero grad optimizer
    optimizer.zero_grad()

    # 6 - BackPropagation
    loss_fn.backward()

    # 7 - Actualizar pesos
    optimizer.step()

    return loss_fn

def test_step(model: torch.nn.Module,
               X,
               y,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):

    # 1 - Modo de inferencia
    model.eval()

    with torch.inference_mode():

        # 2 - Poner los datos en el hardware
        X, y = X.to(device), y.to(device)

        # 3 - Forward Propagation
        y_pred = model(X)
       #print(y_pred)

        # 4 - Calcula perdida
        loss_fn = loss_fn(y_pred, y)

    return loss_fn

def train(model: torch.nn.Module,
          X_train,
          y_train,
          X_test,
          y_test,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          device: torch.device,
          epochs:int):

    for epoch in range(0,epochs):

        loss = train_step(
            model = model,
            X=X_train,
            y=y_train,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )

        if epoch % 100 == 0:
         print(f"Epoch: {epoch}, loss (train): {loss}")

        loss = test_step(
                    model = model,
                    X=X_test,
                    y=y_test,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    device=device
                )
        
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, loss (test): {loss}")