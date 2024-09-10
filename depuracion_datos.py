import pandas as pd
from sklearn.preprocessing import StandardScaler

# Cargar los datos de los tres integrantes
csv_files = ['alex_password.csv', 'Oswaldo_password.csv', 'bolio_password.csv']  # Nombres de los archivos CSV
usuarios = ['alex', 'oswaldo', 'bolio']  # Etiquetas de los usuarios
datos = []  # Para almacenar todos los datos depurados

# Leer y depurar cada archivo CSV
for file, user in zip(csv_files, usuarios):
    # Leer el archivo CSV, saltando la primera fila (nombres de columnas/contraseña) y eliminando la primera columna (iteración)
    df = pd.read_csv(file, skiprows=1).iloc[:, 1:]  # Elimina la primera fila y la primera columna
    
    # Revisar y eliminar filas con valores faltantes
    df.dropna(inplace=True)
    
    # Verificar que todas las filas tengan el número correcto de columnas (24 columnas de tiempos)
    if len(df.columns) != 24:
        print(f"Advertencia: {file} tiene un número inesperado de columnas.")
        continue  # Ajusta según sea necesario

    # Añadir etiqueta del usuario al DataFrame
    df['usuario'] = user  # Añade una columna con la etiqueta del usuario
    datos.append(df)

# Concatenar todos los datos depurados en un solo DataFrame
datos_consolidados = pd.concat(datos, ignore_index=True)

# Normalizar los tiempos
scaler = StandardScaler()
datos_consolidados.iloc[:, :-1] = scaler.fit_transform(datos_consolidados.iloc[:, :-1])  # Normalizar sin incluir la etiqueta

# Mostrar los primeros registros depurados y etiquetados
print(datos_consolidados.head())

# Guardar los datos depurados si es necesario
datos_consolidados.to_csv('datos_cadencia_tecleo.csv', index=False)
