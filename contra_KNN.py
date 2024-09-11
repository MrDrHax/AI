
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Cargar los datos depurados y normalizados
datos = pd.read_csv('datos_cadencia_tecleo.csv')

# Separar características (tiempos) y etiquetas (usuarios)
# Características: todas las columnas excepto la última (etiqueta)
X = datos.iloc[:, :-1].to_numpy(dtype=float)
y = datos['usuario'].to_numpy(dtype=str)     # Etiquetas: última columna

# Dividir los datos en conjunto de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Inicializar el modelo KNN con K-vecinos
knn = KNeighborsClassifier(n_neighbors=3)

# Entrenar el modelo con el conjunto de entrenamiento
knn.fit(X_train, y_train)

# ver si sirve

y_predicted = knn.predict(X_test)


# Create a confusion matrix
cm = confusion_matrix(y_test, y_predicted)

# Normalize the confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(8, 6))

# Create a heatmap of the confusion matrix
heatmap = ax.imshow(cm_normalized, cmap='Blues')

# Add colorbars
fig.colorbar(heatmap)

# Add labels
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')
ax.set_title('Confusion matrix')

# Add the values to the heatmap
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center')

# Add ticks
ax.xaxis.set_ticks(np.arange(cm.shape[1]))
ax.yaxis.set_ticks(np.arange(cm.shape[0]))

# Set tick labels
ax.xaxis.set_ticklabels(['alex', 'bolio', 'oswaldo'])
ax.yaxis.set_ticklabels(['alex', 'bolio', 'oswaldo'])

# Rotate tick labels
plt.xticks(rotation=45)

# Show the plot
plt.show()


# encontre esto oswi. No se si te ayude a hacer lo que querias ver....
knn.kneighbors_graph(X_test)

# # Seleccionar un ejemplo del conjunto de prueba para graficar sus vecinos más cercanos
# punto_prueba = X_test.iloc[0].values.reshape(1, -1)  # Selecciona el primer punto del conjunto de prueba
# punto_prueba_label = y_test.iloc[0]  # Etiqueta real del punto de prueba

# # Obtener los índices de los K vecinos más cercanos
# distancias, indices = knn.kneighbors(punto_prueba)

# # Reducir los datos a 2 dimensiones usando PCA para visualización
# pca = PCA(n_components=2)
# X_train_reducido = pca.fit_transform(X_train)
# punto_prueba_reducido = pca.transform(punto_prueba)

# # Crear la gráfica
# plt.figure(figsize=(10, 6))

# # Graficar los puntos del conjunto de entrenamiento
# plt.scatter(X_train_reducido[:, 0], X_train_reducido[:, 1], c='lightgray', label='Datos de Entrenamiento')

# # Resaltar los K vecinos más cercanos
# plt.scatter(X_train_reducido[indices[0], 0], X_train_reducido[indices[0], 1], c='red', label='K Vecinos Más Cercanos', edgecolor='k', s=100)

# # Graficar el punto de prueba
# plt.scatter(punto_prueba_reducido[0, 0], punto_prueba_reducido[0, 1], c='blue', label=f'Punto de Prueba (Etiqueta Real: {punto_prueba_label})', edgecolor='k', s=150)

# plt.title('Visualización de los K Vecinos Más Cercanos')
# plt.xlabel('Componente Principal 1')
# plt.ylabel('Componente Principal 2')
# plt.legend()
# plt.grid(True)
# plt.show()
