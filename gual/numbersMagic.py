import numpy as np
import matplotlib.pyplot as plt

thresh = 0.5

with open("digitos.txt", 'r') as f:
    # Put the arrays in a list
    array_list = [[] for _ in range(10)]

    for line in f.readlines():
        values = line.replace('\n', '').split(' ')
        leType = int(values[-1]) if values[-1] != '10' else 0

        array_list[leType].append(np.array(values[:-1], dtype=float))

trainingData = [[] for _ in range(10)]


for arr, train in zip(array_list, range(len(trainingData))):
    # Convert the list to a 2D NumPy array
    array_2d = np.array(arr)

    # Calculate the mean along the second axis (axis=0)
    average_array = np.where(np.mean(array_2d, axis=0)
                             < thresh, -1, 1).reshape((20, 20)).T
    trainingData[train] = average_array.flatten()

    # # Scale the values to the range of byte values (0-255)
    # arr_scaled: np.ndarray = (average_array * 255).astype(np.uint8)

    # # Show (debug)
    # plt.imshow(arr_scaled, cmap='gray')
    # plt.show()


def process(array):
    return np.where(array < thresh, -1, 1).reshape((20, 20)).T.flatten()


def plot(array, ax: plt.axes):
    # Scale the values to the range of byte values (0-255)
    arr_scaled: np.ndarray = (array * 255).astype(np.uint8).reshape((20, 20))

    # Show (debug)
    ax.imshow(arr_scaled, cmap='gray')


# configs = [
#     [trainingData[0], trainingData[1]],
#     [trainingData[0], trainingData[3]],
#     [trainingData[2], trainingData[4]],
#     [trainingData[5], trainingData[8]],
#     [trainingData[5], trainingData[6]],
#     [trainingData[1], trainingData[2]],
#     [trainingData[1], trainingData[8]],
#     [trainingData[4], trainingData[7]],
#     [trainingData[9], trainingData[4]],
# ]

# nodes = [
#     [0, 1, 2],
#     [1, -1, -1],
#     [2, 4, 7],
#     [3, 4, 7],
#     [2, 4, 7],
#     [2, 4, 7],
# ]

size = len(trainingData[0])  # int(np.sqrt(len(trainingData[0])))
weights = np.zeros((size, size))

# for entry in trainingData:
#     ledata = entry.reshape(-1, 1)
#     weights += np.dot(ledata, ledata.T)/len(ledata)

for entry in [trainingData[0], trainingData[1]]:
    ledata = entry.reshape(-1, 1)
    weights += np.dot(ledata, ledata.T)/len(ledata)


np.fill_diagonal(weights, 0)


def find(weights: np.ndarray, data: np.ndarray, iterations: int = 10):
    toReturn = data.copy()
    for _ in range(iterations):
        toReturn = np.sign(np.dot(weights, toReturn))

    return toReturn


fig, axs = plt.subplots(3, 10, figsize=(10, 10))

for i in range(10):
    toAnalize = process(array_list[1][i])
    result = find(weights=weights, data=toAnalize)
    plot(toAnalize, axs[0, i])
    plot(result, axs[1, i])

    plot(trainingData[1], axs[2, i])

fig.text(0.5, 0.85, 'Prueba', ha='center', va='center', fontsize=14)
fig.text(0.5, 0.60, 'Reconocido', ha='center', va='center', fontsize=14)
fig.text(0.5, 0.35, 'Patrón esperado', ha='center', va='center', fontsize=14)

plt.show()
