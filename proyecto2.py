import math, random
import pandas as pd

import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+math.e**(-x))


class Neuron:
    weights: list[float]
    _weights: list[float]
    errors: list[float]
    connections: list['Neuron']
    ready: bool
    heated: float
    isStart: bool

    def __init__(self, connections: list['Neuron'], isStart: bool):
        self.connections = connections
        self.weights = [random.random() for _ in range(
            len(self.connections))]
        self.weights.append(random.random())

        self._weights = self.weights.copy()

        self.errors = [0] * len(self.connections)
        self.errors.append(0)

        self.ready = False
        self.isStart = isStart

    def adjustWeights(self, alfa, expected, otherError, startingValues: list[float]):
        value = self.getValue(startingValues, self.isStart)
        if value == expected:
            # self.errors = [0 for _ in self.errors]
            # self._weights = self.weights
            return

        for i in range(len(self.weights) - 1):
            self.errors[i] = (expected-value) * startingValues[i]
            # self.errors[i] = startingValues[i] if self.isStart else self.connections[i].heated\
            #     * (1 - (startingValues[i] if self.isStart else self.connections[i].heated)) * otherError * self.weights[i]
            self._weights[i] = self.weights[i] + alfa * self.errors[i]

        # self.errors[-1] = (expected - value) * otherError * self.weights[-1]
        self.errors[-1] = (expected - value)
        self._weights[-1] = self.weights[-1] + alfa * self.errors[-1]

    def applyWeight(self):
        self.ready = False
        self.weights = self._weights

    def calculateResponse(self, startingValues: list[float], canHeat):
        if self.isStart:
            return sum([self.weights[i] * startingValues[i]
                        for i in range(len(startingValues))]) + self.weights[-1]

        return sum([self.weights[i] * self.connections[i].getValue(startingValues, canHeat) for i in range(len(self.connections))]) + self.weights[-1]

    def getValue(self, startingValues: list[float], canHeat=False):
        # if not self.ready or not canHeat:
        self.heated = self.calculateResponse(startingValues, canHeat)
        # self.ready = True

        return 0 if self.heated < 0.5 else 1

    @staticmethod
    def createModel(layers: list[int], inputs: int) -> list[list['Neuron']]:
        model: list[list[Neuron]] = []

        for i in range(len(layers)):
            model.append([Neuron(model[i - 1] if i != 0 else [0] * inputs,
                         isStart=i == 0,) for j in range(layers[i])])

        return model

    @staticmethod
    def trainModel(model: list[list['Neuron']], alfa: float, repeats: int, trainingSet: list[list[float]], expectedValues: list[list[float]]):
        model.reverse()
        for r in range(repeats):
            print(f'Training, eval:{r}')
            # iterate through layers
            for set in range(len(trainingSet)):
                for j in range(len(model)):
                    for i in range(len(model[j])):
                        model[j][i].adjustWeights(
                            alfa=alfa,
                            # TODO correct for multiple depth
                            expected=expectedValues[set][i] if j == 0 else 0,
                            otherError=1 if j == 0 else model[j - \
                                                              1][i].errors[i],
                            startingValues=trainingSet[set],
                        )
                for j in range(len(model)):
                    for i in range(len(model[j])):
                        model[i][j].applyWeight()

        model.reverse()

    @staticmethod
    def predict(model: list[list['Neuron']], input: list[float]):
        return [i.getValue(input) for i in model[-1]]


df = pd.read_csv("data2.csv", names=['x1', 'x2', 'y'])

x = df[['x1', 'x2']].values.tolist()  
y = df[['y']].values.tolist()           

model = Neuron.createModel([1], 2)
Neuron.trainModel(model, 0.01, 100, x, y)

predictions = [[],[]]

for i in x:
    p= Neuron.predict(model, i)
    predictions[p[0]].append(i)

data1 = predictions[0] 
data2 = predictions[1]

x1, y1 = zip(*[point for point in data1])

x2, y2 = zip(*[point for point in data2])

# Create the scatter plots
plt.figure(figsize=(8, 6))
plt.scatter(x1, y1, color='red', label='0')  # First scatter plot
plt.scatter(x2, y2, color='green', label='1')   # Second scatter plot

plt.grid()

# Show the plot
plt.show()
