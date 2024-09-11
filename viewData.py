import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
import hashlib

# Cargar los datos de los tres integrantes
csv_files = ['alex_password.csv', 'Oswaldo_password.csv',
             'bolio_password.csv']  # Nombres de los archivos CSV
usuarios = ['alex', 'oswaldo', 'bolio']  # Etiquetas de los usuarios


def string_to_color(s, offset = 0):
    # Hash the string to get a 6-digit hexadecimal number
    h = hashlib.sha256(s.encode()).hexdigest()[:6]

    # Convert the hexadecimal number to a color code
    r = int(h[:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:], 16)
    color = '#{:02x}{:02x}{:02x}'.format(max(r - offset, 0), max(g - offset, 0 ), max(b - offset, 0))

    return color

x = []
tag = []
colors = []

# Leer y depurar cada archivo CSV
for file, user in zip(csv_files, usuarios):
    # Leer el archivo CSV, agregando indices y headers existentes
    df = pd.read_csv(file, dtype=float, index_col=0)
    df = df.iloc[:, 1:]

    metrics = df.describe()

    mean = metrics.loc['mean'].to_numpy()
    std = metrics.loc['std'].to_numpy()

    x.append(mean)
    tag.append(f'{user} - mean')
    colors.append(string_to_color(user))

    x.append(mean - std)
    x.append(mean + std)
    tag.append(f'{user} - std min')
    tag.append(f'{user} - std max')
    colors.append(string_to_color(user, 100))
    colors.append(string_to_color(user, 100))



# Create a new figure
fig, ax = plt.subplots()

# Plot each line using the data parameter
for i in range(len(x)):
    ax.plot(x[i], data=ax, label=tag[i], color=colors[i])

# Add a legend
ax.legend()

# Show the plot
plt.show()
