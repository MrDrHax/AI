import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def graficaDatos(x, y, theta):
    # agregar datos al plot, y crear una linea
    plt.scatter(x, y, color='blue')
    plt.plot(x, x*theta[1] + theta[0], color="red")

    plt.show()


def calculaCosto(x: np.ndarray, y: np.ndarray, theta=[0, 0]):
    # calcular el error, usandouna matriz
    return np.sum(np.square((theta[0] + theta[1] * x[:, 1]) - y))


def gradienteDescendiente(x: np.ndarray, y: np.ndarray, theta=[0, 0], alpha=10, depth=1000):
    # calcula error inicial (para no empeorar)
    error_p = calculaCosto(x, y, theta)

    # se itera n cantidad de veces
    for i in range(depth):
        # se calcula nuevo theta
        t0 = theta[0] - alpha * (np.sum(x.dot(theta)-y)) / (2*len(x))
        t1 = theta[1] - alpha * (np.sum((x.dot(theta)-y)*x[:, 1])) / (2*len(x))

        # se calcula en nuevo costo
        error = calculaCosto(x, y, [t0, t1])

        # si el costo es menor, theta es bueno
        if error < error_p:
            theta = [t0, t1]
            error_p = error
        # aplicar theta reducido integilentemente
        alpha *= 0.95
        alpha = max(0.001, alpha)

    # al terminar regresar el resultado
    return theta


if __name__ == '__main__':
    # leer archivo
    df = pd.read_csv("data.csv", names=['x', 'y'])

    # Formar datos
    X = np.c_[np.ones(df.shape[0]), df['x']]
    Y = np.array(df['y'])

    # entrenar el modelo
    theta = gradienteDescendiente(X, Y)

    # graficar datos (removiendo inicio)
    graficaDatos(X[:, 1], Y, theta)

    # comprobar
    print(theta)
