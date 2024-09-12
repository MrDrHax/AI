from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Cargar los datos depurados y normalizados
datos = pd.read_csv('datos_cadencia_tecleo.csv')

# Separar características (tiempos) y etiquetas (usuarios)
X = datos.iloc[:, :-1].to_numpy(dtype=float)
y = datos['usuario'].to_numpy(dtype=str)

# Dividir los datos en conjunto de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Obtener las clases únicas para etiquetar la matriz de confusión
classes = np.unique(y)

for k in range(1, 16):
    # Inicializar el modelo KNN con K-vecinos
    knn = KNeighborsClassifier(n_neighbors=k)

    # Entrenar el modelo con el conjunto de entrenamiento
    knn.fit(X_train, y_train)

    # Predecir con el conjunto de prueba
    y_predicted = knn.predict(X_test)

    # Crear la matriz de confusión
    cm = confusion_matrix(y_test, y_predicted)

    # Normalizar la matriz de confusión, manejando posibles divisiones por cero
    with np.errstate(all='ignore'):
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # Reemplazar NaN por 0

    # Configurar la figura de matplotlib
    fig, ax = plt.subplots(figsize=(8, 6))

    # Crear un mapa de calor de la matriz de confusión
    heatmap = ax.imshow(cm_normalized, cmap='Blues')

    # Añadir barras de color
    fig.colorbar(heatmap)

    # Añadir etiquetas y título
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(f'Confusion Matrix for k = {k}')

    # Añadir los valores a la matriz de calor
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='black')

    # Configurar los ticks y sus etiquetas
    ax.xaxis.set_ticks(np.arange(len(classes)))
    ax.yaxis.set_ticks(np.arange(len(classes)))
    ax.xaxis.set_ticklabels(classes)
    ax.yaxis.set_ticklabels(classes)

    # Rotar las etiquetas del eje x
    plt.xticks(rotation=45)

    # Guardar y cerrar la figura para evitar problemas de memoria
    plt.savefig(f'img/cm_{k}.png')
    plt.close(fig)

    # Mostrar las métricas de evaluación
    print(
        f'Metrics k = {k}:\n'
        f'- accuracy: {accuracy_score(y_test, y_predicted):.2f}\n'
        f'- precision: {precision_score(y_test, y_predicted, average="macro"):.2f}\n'
        f'- recall: {recall_score(y_test, y_predicted, average="macro"):.2f}\n'
        f'- f1: {f1_score(y_test, y_predicted, average="macro"):.2f}\n'
    )

# Si no se va a utilizar, se puede eliminar esta línea
# knn.kneighbors_graph(X_test)
