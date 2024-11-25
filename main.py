# Importamos las librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ==============================================
# FUNCIÓN 1: Regresión Lineal
# ==============================================
def regresion_lineal():
    """
    Función que implementa la regresión lineal para predecir el precio
    de una casa basado en los metros cuadrados.
    """
    # Generamos datos simulados
    np.random.seed(42)
    metros_cuadrados = np.random.randint(50, 300, 100)
    precios = metros_cuadrados * 1500 + np.random.normal(0, 20000, 100)

    # Dividimos los datos en conjuntos de entrenamiento y prueba
    X = metros_cuadrados.reshape(-1, 1)
    y = precios
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creamos y entrenamos el modelo de regresión lineal
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # Hacemos predicciones
    y_pred = modelo.predict(X_test)

    # Calculamos el error cuadrático medio
    mse = mean_squared_error(y_test, y_pred)
    print(f"Error cuadrático medio (Regresión Lineal): {mse}")

    # Visualizamos los resultados
    plt.figure(figsize=(10, 5))
    plt.scatter(X, y, color='blue', label="Datos originales")
    plt.plot(X, modelo.predict(X), color='red', label="Regresión Lineal")
    plt.xlabel("Metros cuadrados")
    plt.ylabel("Precio de la casa")
    plt.legend()
    plt.title("Regresión Lineal: Precio de una Casa")
    plt.show()

# ==============================================
# FUNCIÓN 2: Regresión Polinómica
# ==============================================
def regresion_polinomica():
    """
    Función que implementa la regresión polinómica para predecir el
    rendimiento de un cultivo basado en la cantidad de fertilizante.
    """
    # Generamos datos simulados
    np.random.seed(42)
    fertilizante = np.linspace(0, 100, 100)
    rendimiento = 100 - 0.5 * (fertilizante - 50) ** 2 + np.random.normal(0, 5, 100)

    # Dividimos los datos en conjuntos de entrenamiento y prueba
    X = fertilizante.reshape(-1, 1)
    y = rendimiento
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Transformamos los datos a características polinómicas
    polynomial_features = PolynomialFeatures(degree=2)
    X_poly_train = polynomial_features.fit_transform(X_train)
    X_poly_test = polynomial_features.transform(X_test)

    # Creamos y entrenamos el modelo de regresión polinómica
    modelo = LinearRegression()
    modelo.fit(X_poly_train, y_train)

    # Hacemos predicciones
    y_pred = modelo.predict(X_poly_test)

    # Calculamos el error cuadrático medio
    mse = mean_squared_error(y_test, y_pred)
    print(f"Error cuadrático medio (Regresión Polinómica): {mse}")

    # Visualizamos los resultados
    plt.figure(figsize=(10, 5))
    plt.scatter(fertilizante, rendimiento, color='blue', label="Datos originales")
    plt.plot(fertilizante, modelo.predict(polynomial_features.transform(X)), color='red', label="Regresión Polinómica")
    plt.xlabel("Cantidad de Fertilizante")
    plt.ylabel("Rendimiento del Cultivo")
    plt.legend()
    plt.title("Regresión Polinómica: Rendimiento del Cultivo")
    plt.show()

# ==============================================
# EJECUCIÓN DE LAS FUNCIONES
# ==============================================
if __name__ == "__main__":
    print("Ejecutando regresión lineal...")
    regresion_lineal()

    print("\nEjecutando regresión polinómica...")
    regresion_polinomica()
