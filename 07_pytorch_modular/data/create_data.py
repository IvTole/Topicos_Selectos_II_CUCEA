import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt

    # Scikit learn
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_digits

    return load_digits, pd, plt, train_test_split


@app.cell
def _(load_digits, pd):
    # Importación de datos

    digits = load_digits()

    # Features
    X = digits.data

    # Target
    y = digits.target

    df = pd.DataFrame(X, columns=[f"pix_{i}" for i in range(X.shape[1])])
    df["target"] = y

    df.head()
    return (df,)


@app.cell
def _(df, plt):
    image = df.iloc[0][:-1].to_numpy().reshape(8,8) # todas las características menos el target
    print(image)

    # plot
    plt.imshow(image, cmap="grey")
    plt.show()
    return


@app.cell
def _(df, train_test_split):
    # split de datos entrenamiento y prueba
    df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=42, stratify=df["target"])

    print(f"shape (train): {len(df_train)}")
    print(f"shape (test): {len(df_test)}")

    df_train["target"].value_counts()
    return df_test, df_train


@app.cell
def _(df_test, df_train):
    # Exportación
    df_train.to_csv("train.csv", header=True, index=None)
    df_test.to_csv("test.csv", header=True, index=None)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
