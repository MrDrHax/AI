from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Cargar los datos depurados y normalizados
datos = pd.read_csv('datos_cadencia_tecleo.csv')

# Separar características (tiempos) y etiquetas (usuarios)
X = datos.iloc[:, :-1].to_numpy(dtype=float)
y = datos['usuario'].to_numpy(dtype=str)

# Dividir los datos en conjunto de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar el modelo KNN con el mejor k encontrado previamente (10)
knn = KNeighborsClassifier(n_neighbors=10)

# Entrenar el modelo
knn.fit(X_train, y_train)

# Obtener las probabilidades de predicción
y_probs = knn.predict_proba(X_test)

# Lista para almacenar los resultados
classes = knn.classes_

# Colores personalizados para cada clase
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Colores distintos para cada clase

# Generar la curva de ROC para cada clase
plt.figure(figsize=(10, 8))

# Generar la curva de ROC para cada clase
for i, (class_name, color) in enumerate(zip(classes, colors)):
    # Convertir etiquetas a valores binarios: 1 si es la clase de interés, 0 si no
    y_binary = (y_test == class_name).astype(int)
    
    # Obtener las probabilidades para la clase de interés
    probs_class = y_probs[:, i]
    
    # Calcular la tasa de falsos positivos y verdaderos positivos
    fpr, tpr, _ = roc_curve(y_binary, probs_class)
    plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc_score(y_binary, probs_class):0.2f})")

# Añadir línea horizontal en y=0.5 para resaltar el umbral
plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)

# Configuración del gráfico
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC AUC Curves for Each User', fontsize=16)
plt.legend(loc='best', fontsize=12, frameon=True, shadow=True, facecolor='white', edgecolor='gray')
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()

# Mostrar el gráfico
plt.show()