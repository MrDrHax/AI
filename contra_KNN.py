import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Cargar los datos depurados y normalizados
datos = pd.read_csv('depuracion_datos.csv')

# Separar características (tiempos) y etiquetas (usuarios)
X = datos.iloc[:, :-1]  # Características: todas las columnas excepto la última (etiqueta)
y = datos['usuario']    # Etiquetas: última columna

# Dividir los datos en conjunto de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar el modelo KNN con K vecinos
knn = KNeighborsClassifier(n_neighbors=3)

# Entrenar el modelo con el conjunto de entrenamiento
knn.fit(X_train, y_train)

# Seleccionar un ejemplo del conjunto de prueba para graficar sus vecinos más cercanos
punto_prueba = X_test.iloc[0].values.reshape(1, -1)  # Selecciona el primer punto del conjunto de prueba
punto_prueba_label = y_test.iloc[0]  # Etiqueta real del punto de prueba

# Obtener los índices de los K vecinos más cercanos
distancias, indices = knn.kneighbors(punto_prueba)

# Reducir los datos a 2 dimensiones usando PCA para visualización
pca = PCA(n_components=2)
X_train_reducido = pca.fit_transform(X_train)
punto_prueba_reducido = pca.transform(punto_prueba)

# Crear la gráfica
plt.figure(figsize=(10, 6))

# Graficar los puntos del conjunto de entrenamiento
plt.scatter(X_train_reducido[:, 0], X_train_reducido[:, 1], c='lightgray', label='Datos de Entrenamiento')

# Resaltar los K vecinos más cercanos
plt.scatter(X_train_reducido[indices[0], 0], X_train_reducido[indices[0], 1], c='red', label='K Vecinos Más Cercanos', edgecolor='k', s=100)

# Graficar el punto de prueba
plt.scatter(punto_prueba_reducido[0, 0], punto_prueba_reducido[0, 1], c='blue', label=f'Punto de Prueba (Etiqueta Real: {punto_prueba_label})', edgecolor='k', s=150)

plt.title('Visualización de los K Vecinos Más Cercanos')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend()
plt.grid(True)
plt.show()
