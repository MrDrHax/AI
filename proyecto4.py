from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random


def calculateDistance(a: np.ndarray, b: np.ndarray):
    return np.linalg.norm(a-b)


debugsX = [[], [], [], []]
debugsY = [[], [], [], []]


def calculateMeans(dataPoints: list[np.ndarray], k: int, epochs: int):
    minVal = np.min(dataPoints, axis=0)
    maxVal = np.max(dataPoints, axis=0)
    means = np.array([[random.uniform(minVal[i], maxVal[i])
                      for i in range(len(minVal))] for _ in range(k)])

    tags = [0] * len(dataPoints)

    for i in range(epochs):
        # get distances
        for point in range(len(dataPoints)):
            distances = [calculateDistance(
                dataPoints[point], mean) for mean in means]
            index = distances.index(min(distances))

            tags[point] = index

        averages = [np.zeros(len(dataPoints[0])) for _ in range(k)]
        totals = [0] * k

        for j in range(len(dataPoints)):
            averages[tags[j]] += dataPoints[tags[j]]
            totals[tags[j]] += 1

        for j in range(len(averages)):
            if totals[j] == 0:
                averages[j] = means[j].copy()
                continue
            for z in range(len(averages[j])):
                averages[j][z] /= totals[j]

        # # for m in range(len(means)):
        # debugsX[0].append(means[0][0])
        # debugsX[1].append(means[1][0])
        # debugsX[2].append(means[2][0])
        # debugsX[3].append(means[3][0])
        # debugsY[0].append(means[0][1])
        # debugsY[1].append(means[1][1])
        # debugsY[2].append(means[2][1])
        # debugsY[3].append(means[3][1])
        # plotData(np.array(means), np.array(tags), np.array(dataPoints))

        means = averages.copy()

    return means, tags


def plotData(means, tags, data):
    cmap = plt.get_cmap('Set3')

    norm = plt.Normalize(min(tags), max(tags))

    plt.scatter(data[:, 0], data[:, 1],
                c=[cmap(norm(t)) for t in tags])

    plt.scatter(means[:, 0], means[:, 1], c='red')

    # plt.plot(debugsX[0], debugsY[0], 'bo--')
    # plt.plot(debugsX[1], debugsY[1], 'bo--')
    # plt.plot(debugsX[2], debugsY[2], 'bo--')
    # plt.plot(debugsX[3], debugsY[3], 'bo--')

    # Show the plot
    plt.show()


# datapoints = np.array([[random.uniform(1, 10)
#                        for _ in range(2)] for _ in range(10)])


# Load the PNG image
image_path = 'birb.png'  
image = Image.open(image_path)

# Convert the image to RGB (if it's not already in that mode)
image = image.convert('RGB')

# Get the pixel data
pixels = np.array(image)

# Reshape the pixel data to the desired format
# Each pixel is represented as [R, G, B]
pixel_array = pixels.reshape(-1, 3)

means, tags = calculateMeans(pixel_array, 16, 100)

# plotData(np.array(means), np.array(tags), np.array(datapoints))

# convert back into pixels
remade_pixels = []
for tag in tags:
    remade_pixels.append(means[tag])

remade_pixels = np.array(remade_pixels)


image_width = 128
image_height = 128
reshaped_pixels = remade_pixels.reshape((image_height, image_width, 3))

# Create an image from the reshaped pixel array
image = Image.fromarray(reshaped_pixels.astype('uint8'), 'RGB')

# Save the image or show it
image.save('output_image.png')  # Save the image
image.show()
