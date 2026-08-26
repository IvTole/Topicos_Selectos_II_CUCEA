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

    return ColumnTransformer, StandardScaler, mo, np, pd, plt


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
    df_filter = df[["HorsePower","Cars Prices"]]
    df_filter
    return (df_filter,)


@app.cell
def _(ColumnTransformer, StandardScaler, df_filter):
    ## Pipeline
    num_cols = ["HorsePower", "Cars Prices"]

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

    df_processed.columns = ["HorsePower", "Cars Prices"]

    df_processed
    return (df_processed,)


@app.cell
def _(df_processed, plt):
    plt.scatter(df_processed["HorsePower"], df_processed["Cars Prices"])
    return


@app.cell
def _(df_processed, np):
    # Pasar este dataframe a arreglos de numpy
    X_std = df_processed["HorsePower"]
    y_std = df_processed["Cars Prices"]

    X_std = np.array(X_std).reshape((1, len(X_std)))
    y_std = np.array(y_std).reshape((1, len(y_std)))

    print(f"X_std (shape): {str(X_std.shape)}")
    print(f"y_std (shape): {str(y_std.shape)}")
    return X_std, y_std


@app.cell(hide_code=True)
def _():
    return


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
        W = np.random.randn(n_x, n_y)*0.1

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
def _():
    return


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

        print(((y_hat - y)**2.0))

        # cálculo del error (pérdida)
        loss = np.sum((y_hat - y)**2.0) / (2*m)

        return loss

    return (loss_error,)


@app.cell
def _(loss_error, y_hat, y_std):
    loss = loss_error(y_hat=y_hat, y=y_std)
    print(f"error: {loss}")
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
