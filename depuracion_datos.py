import pandas as pd
from sklearn.preprocessing import StandardScaler

# Cargar los datos de los tres integrantes
csv_files = ['alex_password.csv', 'Oswaldo_password.csv', 'bolio_password.csv']  # Nombres de los archivos CSV
usuarios = ['alex', 'oswaldo', 'bolio']  # Etiquetas de los usuarios
datos = []  # Para almacenar todos los datos depurados


# Leer y depurar cada archivo CSV
for file, user in zip(csv_files, usuarios):
    # Leer el archivo CSV, agregando indices y headers existentes
    df = pd.read_csv(file, dtype=float, index_col=0)

    df = df.iloc[:, 1:]  # remover primer columna, puesto que no se sabe cuanto tiempo se tarda en iniciar a escribir password el usuario
    
    # Verificar que todas las filas tengan el número correcto de columnas (24 columnas de tiempos)
    if len(df.axes[1]) != 23:
        print(f"Advertencia: {file} tiene un número inesperado de columnas.")
        continue  # Ajusta según sea necesario

    if len(df.axes[0]) != 50:
        print(f"Advertencia: {file} tiene un número inesperado de filas.")
        continue  # Ajusta según sea necesario

    # Añadir etiqueta del usuario al DataFrame
    df['usuario'] = user  # Añade una columna con la etiqueta del usuario
    datos.append(df)

mixData = pd.concat(datos, ignore_index=True)
mixData.to_csv('datos_cadencia_tecleo.csv', index=False)
