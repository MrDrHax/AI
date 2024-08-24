import matplotlib.pyplot as plt
import numpy as np
import random


def calculateDistance(a: np.ndarray, b: np.ndarray):
    return np.linalg.norm(a-b)


def calculateMeans(dataPoints: list[np.ndarray], k: int, epochs: int):
    minVal = np.min(dataPoints, axis=0)
    maxVal = np.max(dataPoints, axis=0)
    means = [np.array([random.uniform(minVal[i], maxVal[i])
                      for i in range(len(minVal))]) for _ in range(k)]

    tags = [0] * len(dataPoints)

    for i in range(epochs):
        # get distances
        for point in range(len(dataPoints)):
            distances = [calculateDistance(
                dataPoints[point], mean) for mean in means]
            index = distances.index(max(distances))

            tags[point] = index

        averages = [np.zeros(len(dataPoints[0])) for _ in range(k)]
        totals = [0] * k

        for j in range(len(dataPoints)):
            averages[tags[j]] += dataPoints[tags[j]]
            totals[tags[j]] += 1

        for j in range(len(averages)):
            averages[j] /= totals[j]

        means = averages

    return means, tags


def plotData(means, tags, data):
    cmap = plt.get_cmap('Set3')

    norm = plt.Normalize(min(tags), max(tags))

    plt.scatter(data[:, 0], data[:, 1],
                c=[cmap(norm(t)) for t in tags])

    plt.scatter(means[:, 0], means[:, 1], c='red')

    # Show the plot
    plt.show()


datapoints = [np.array([random.uniform(1, 10)
                       for _ in range(2)]) for _ in range(20)]

means, tags = calculateMeans(datapoints, 4, 100)

print(tags)
print(means)

plotData(np.array(means), np.array(tags), np.array(datapoints))
