import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer

    from sklearn.linear_model import LinearRegression

    return ColumnTransformer, LinearRegression, StandardScaler, mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Perceptron (regresión)

    ![Perceptron](images/perceptron.png)
    """)
    return


@app.cell
def _(pd):
    ## Importacion de datos
    df = pd.read_csv("../data/cars/cars_2025.csv", encoding='latin-1')

    # Nombres de columnas
    df = df.rename(columns={"Performance(0 - 100 )KM/H":"Performance"})

    ## "limpieza"

    # Extract numeric HP
    df['HorsePower'] = df['HorsePower'].str.extract(r'(\d+)').astype(float)

    # Extract numeric Top Speed (km/h)
    df['Total Speed'] = df['Total Speed'].str.extract(r'(\d+)').astype(float)

    # Extract numeric Acceleration (seconds)
    df['Performance'] = df['Performance'].str.extract(r'([\d.]+)').astype(float)

    # Extract numeric Price (USD)
    df['Cars Prices'] = df['Cars Prices'].str.replace('[$,]', '', regex=True)\
                                  .str.extract(r'(\d+)').astype(float)

    # Standardize Company names to Title Case
    df['Company Names'] = df['Company Names'].str.strip().str.title()
    df['Fuel Types'] = df['Fuel Types'].str.strip().str.title()

    df = df.dropna()

    df.head()
    return (df,)


@app.cell
def _(df, plt):
    plt.scatter(df["HorsePower"], df["Cars Prices"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preprocessing pipeline
    """)
    return


@app.cell
def _(df):
    df_filter = df[["Performance","Total Speed","HorsePower","Cars Prices"]]
    df_filter
    return (df_filter,)


@app.cell
def _(ColumnTransformer, StandardScaler, df_filter):
    ## Pipeline
    num_cols = ["Performance","Total Speed","HorsePower", "Cars Prices"]

    ## Scaler
    scaler = StandardScaler()

    # ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', scaler, num_cols)
        ],
        remainder="passthrough"
    )

    preprocessor.set_output(transform="pandas")

    # Aplicar el proceso

    # Fitting
    preprocessor_fitted = preprocessor.fit(df_filter)

    # Transform
    df_processed = preprocessor_fitted.transform(df_filter)

    df_processed.columns = ["Performance","Total Speed", "HorsePower", "Cars Prices"]

    df_processed
    return (df_processed,)


@app.cell
def _(df_processed, plt):
    plt.scatter(df_processed["HorsePower"], df_processed["Cars Prices"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Convertimos a un array

    Transformamos las variables en arreglos de numpy, matrices.
    """)
    return


@app.cell
def _(df_processed, np):
    # Pasar este dataframe a arreglos de numpy
    X = df_processed[["Performance","Total Speed","HorsePower"]]
    y = df_processed["Cars Prices"]

    X_std = np.array(X).T
    y_std = np.array(y).reshape((1, len(y)))

    print(f"X_std (shape): {str(X_std.shape)}")
    print(f"y_std (shape): {str(y_std.shape)}")
    return X, X_std, y, y_std


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Implementación de una red neuronal (modelo de regresion lineal)

    ### Paso 1

    Definimos dimensiones del perceptron
    """)
    return


@app.function
def layer_sizes(X,y):
    """
    Argumentos:
    X - conjunto de datos de entrada con dimensiones (tamaño de entrada, numero de datos(observaciones))
    y - etiquetas (target) con dimension (tamaño de salida, numero de datos(observaciones))

    Returns (tupla)
    n_x -- tamaño de la capa de entrada
    n_y -- tamaño de la capa de salida
    """

    n_x = X.shape[0]
    n_y = y.shape[0]
    
    return (n_x, n_y)


@app.cell
def _(X_std, y_std):
    (n_x, n_y) = layer_sizes(X=X_std, y=y_std)
    print(f"Tamaño de la capa de entrada: {n_x}")
    print(f"Tamaño de la capa de salida {n_y}")
    return n_x, n_y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Paso 2

    Inicializar pesos (weights) $w$ y sesgo (bias) $b$ de manera aleatoria.
    """)
    return


@app.cell
def _(np):
    def initialize_parameters(n_x, n_y):
        """
        Argumentos:
        Los tamaños de entrada y salida n_x y n_y

        Returns:
        arreglo de pesos w
        arreglo de sesgos b
        """

        # Inicializar aleatoriamente los w's
        W = np.random.randn(n_y, n_x)*0.1

        # Inicializar b's
        b = np.random.randn(n_y, 1)*0.1

        parameters = {
            "W":W,
            "b":b
        }

        return parameters

    return (initialize_parameters,)


@app.cell
def _(initialize_parameters, n_x, n_y):
    parameters = initialize_parameters(n_x=n_x, n_y=n_y)
    parameters
    return (parameters,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Paso 3

    Forward propagation (inferencia)
    """)
    return


@app.cell
def _(np):
    def forward_propagation(X, parameters):

        W = parameters["W"]
        b = parameters["b"]

        # propagacion hacia adelante
        Z = np.matmul(W,X) + b

        # prediccion (inferencia)
        y_hat = Z

        return y_hat


    return (forward_propagation,)


@app.cell
def _(X_std, forward_propagation, parameters):
    y_hat = forward_propagation(X=X_std, parameters=parameters)
    y_hat
    return (y_hat,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Paso 4
    Calcular el error (función de pérdida, Loss Function)

    La función de costo a ser utilizada por el modelo es la siguiente:

    \begin{equation}
    \mathcal{L} (w,b) = \frac{1}{2m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2
    \end{equation}
    """)
    return


@app.cell
def _(np):
    def loss_error(y_hat, y):
        """
        Argumentos:
        y_hat: prediccion de la red neuronal
        y: target (datos)
        """

        # número de observaciones
        m = y_hat.shape[1]

        # cálculo del error (pérdida)
        loss = np.sum((y_hat - y)**2.0) / (2*m)

        return loss

    return (loss_error,)


@app.cell
def _(loss_error, y_hat, y_std):
    loss = loss_error(y_hat=y_hat, y=y_std)
    print(f"error: {loss}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Paso 5 - BackPropagation

    \begin{equation}
    \nabla = (\partial / \partial w, \partial / \partial b)
    \end{equation}

    \begin{equation}
    \nabla \mathcal{L} = (\partial \mathcal{L} / \partial w, \partial \mathcal{L} / \partial b)
    \end{equation}

    Ahora calculamos las derivadas parciales mostradas en secciones anteriores,

    \begin{equation}
    \frac{\partial \mathcal{L}}{\partial w} = \frac{1}{m} \sum_{i=1}^m \left( \hat{y}^{(i)} - y^{(i)} \right) x^{(i)} = \frac{1}{m} \sum_{i=1}^m z^{(i)}x^{(i)} = \frac1m z \cdot x
    \end{equation}

    \begin{equation}
    \frac{\partial \mathcal{L}}{\partial b} = \frac{1}{m} \sum_{i=1}^m \left( \hat{y}^{(i)} - y^{(i)} \right)
    \end{equation}

    Tomamos $z = \hat{y}^{(i)} - y^{(i)}$
    """)
    return


@app.cell
def _(np):
    def back_propagation(y_hat, y, X):
        """
        Argumentos:
        y_hat -- prediccion
        y -- observaciones (reales, datos de entrenamiento)
        X -- matriz de datos (features)

        Retorna:
        grads (diccionario) -- gradiente de la funcion de pérdida L para el punto correspondiente
        """

        # m, número total de observaciones
        m = X.shape[1]

        # Propagacion hacia adelante, derivadas parciales
        dZ = y_hat - y
        dW = (1/m) * np.dot(dZ, X.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)

        # gradiente
        grads = {
            "dW":dW,
            "db":db
        }

        return grads

    return (back_propagation,)


@app.cell
def _(X_std, back_propagation, y_hat, y_std):
    ## gradiente
    grads = back_propagation(y_hat=y_hat, y=y_std, X=X_std)
    grads
    return (grads,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ahora continuamos con el ajuste de los pesos $w$ y el bias $b$ con el descenso del gradiente,

    \begin{equation}
    w = w - \alpha \frac{\partial \mathcal{L}}{\partial w}
    \end{equation}

    \begin{equation}
    b = b - \alpha \frac{\partial \mathcal{L}}{\partial b}
    \end{equation}
    """)
    return


@app.function
def update_parameters(parameters, grads, learning_rate=1.0):
    """
    Argumentos:
    parameters - dict de parametros (W,b)
    grads - dict de gradientes (dW, db)
    learning_rate -- cte para modular el aprendizaje (del descenso del gradiente)

    Retorna:
    updated parameters -- dict (W,b) updated
    """

    # Extraigo parametros
    W = parameters["W"]
    b = parameters["b"]

    # Extraigo gradientes
    dW = grads["dW"]
    db = grads["db"]

    # Método de optimización  (descenso del gradiente)
    W = W - learning_rate * dW
    b = b - learning_rate * db

    parameters = {
        'W':W,
        'b':b
    }

    return parameters


@app.cell
def _(grads, parameters):
    updated_parameters = update_parameters(parameters=parameters, grads=grads, learning_rate=0.5)
    updated_parameters
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Entrenamiento de un perceptron
    """)
    return


@app.cell
def _(
    X_std,
    back_propagation,
    forward_propagation,
    initialize_parameters,
    loss_error,
    y_std,
):
    def nn_model(X, y, iterations=10, learning_rate=1.0, print_cost=False):

        # paso 1 - Definir dimensiones
        (n_x, n_y) = layer_sizes(X=X_std, y=y_std)

        # paso 2 - Inicializar pesos (aleatoriamente)
        parameters = initialize_parameters(n_x=n_x, n_y=n_y)

        loss_list = []
        iter_list = []

        # Ciclo de entrenamiento
        for i in range(0,iterations):

            # paso 3 - Forward Propagation (inferencia)
            y_hat = forward_propagation(X=X_std, parameters=parameters)
    
            # paso 4 - Calcular error (pérdida/loss)
            loss = loss_error(y_hat=y_hat, y=y_std)
    
            # paso 5 - Cálculo del gradiente de la función de pérdida L en esa configuracion
            grads = back_propagation(y_hat=y_hat, y=y_std, X=X_std)
    
            # paso 6 - Optimización y actualizacion de w,b (descenso del gradiente)
            parameters = update_parameters(parameters=parameters, grads=grads, learning_rate=0.5)

            loss_list.append(loss)
            iter_list.append(i+1)

            if print_cost:
                print(f"Pérdida/Loss después de la iteración {i}: {round(loss,5)}")
    
        return parameters, loss_list, iter_list

    return (nn_model,)


@app.cell
def _(X_std, nn_model, y_std):
    parameters_regression, loss_list, iter_list = nn_model(X=X_std, y=y_std, iterations=100, learning_rate=0.8, print_cost=False)
    parameters_regression
    return iter_list, loss_list


@app.cell
def _(iter_list, loss_list, plt):
    plt.plot(iter_list, loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    return


@app.cell
def _(LinearRegression, X, y):
    # Comparativa con regresion lineal tradicional

    # Paso 1, definir X (features), y(target)

    # Paso 2, instanciar el modelo
    model = LinearRegression()

    # Paso 3, entrenamiento
    model.fit(X,y)

    print(f"beta 0 -- b: {model.intercept_}")
    print(f"betas -- w: {model.coef_}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
