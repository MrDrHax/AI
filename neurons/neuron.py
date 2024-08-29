import random
import math
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Neuron:
    weights: np.ndarray
    _weights: np.ndarray
    errors: np.ndarray
    connections: list['Neuron']
    heated: float
    isStart: bool

    def __init__(self, connections: list['Neuron'], isStart: bool):
        self.connections = connections
        # self.weights = [random.random() for _ in range(
        #     len(self.connections) + 1)]

        if isStart:
            self.weights = np.ones(len(self.connections) + 1)
        else:
            self.weights = np.ones(len(self.connections) + 1)

        self._weights = self.weights.copy()

        self.errors = np.zeros(len(self.connections) + 1)

        self.isStart = isStart

        self.heated = None

    def adjustWeights(self, learningRate: float, expected: np.ndarray, startingValues: np.ndarray):
        # Calculate the error for the output layer
        if self.isStart:
            # For input layer, we don't calculate error
            return

        value = self.getValue(startingValues)

        # Calculate the error for the output layer
        self.errors = np.dot(self.weights.T, (value - expected))

        self._weights -= learningRate * self.errors

    def applyWeight(self):
        self.heated = None
        self.weights = self._weights
        pass

    def calculateResponse(self, startingValues: np.ndarray):
        if self.heated is not None:
            return self.heated
        
        if self.isStart:
            Y = np.insert(startingValues, 0, 1)  # add a 1 at the beggining
            self.heated = sigmoid(np.dot(self.weights, Y))
            return self.heated

        # self.heated = sum([self.weights[i] * self.connections[i].getValue(startingValues)
        #                   for i in range(len(self.connections))]) + self.weights[-1]
        self.heated = sigmoid(
            np.dot(self.weights, np.insert(np.array([conn.getValue(startingValues) for conn in self.connections]), 0, 1)))
        return self.heated

    def getValue(self, startingValues: list[float]):
        self.calculateResponse(startingValues)

        return self.heated

    @staticmethod
    def createModel(layers: list[int], inputs: int) -> list[list['Neuron']]:
        model: list[list[Neuron]] = []

        for i in range(len(layers)):
            model.append([Neuron(model[i - 1] if i != 0 else [0] * inputs,
                         isStart=i == 0,) for j in range(layers[i])])

        return model

    @staticmethod
    def trainModel(model: list[list['Neuron']], alfa: float, repeats: int, trainingSet: list[list[float]], expectedValues: list[list[float]]):
        for r in range(repeats):
            print(f'\rTraining, epoch: {r}', end='')
            # iterate through layers
            for set in range(len(trainingSet)):
                # Forward pass
                for j in range(len(model)):
                    for i in range(len(model[j])):
                        # Calculate output for each neuron
                        model[j][i].getValue(trainingSet[set])

                # Backward pass
                for j in range(len(model) - 1, -1, -1):  # Iterate backwards through layers
                    for i in range(len(model[j])):
                        # If not the output layer, calculate hidden layer errors
                        # if j < len(model) - 1:
                        #     model[j][i].calculateHiddenLayerErrors()
                        expected = expectedValues[set][i] if j == len(
                            model) - 1 else sigmoid(sum([model[j + 1][m].errors[i] for m in range(len(model[j+1]))]))
                        model[j][i].adjustWeights(
                            learningRate=alfa,
                            expected=expected,
                            startingValues=trainingSet[set],
                        )

            for j in range(len(model)):
                for i in range(len(model[j])):
                    model[j][i].applyWeight()

        print('\n Finished!!!')

    @staticmethod
    def predict(model: list[list['Neuron']], input: list[float]):
        return [i.getValue(input) for i in model[-1]]


if __name__ == '__main__':
    model = Neuron.createModel([2, 2, 1], 2)
    Neuron.trainModel(
        model=model,
        alfa=0.01,
        repeats=1000,
        trainingSet=[
            np.array([0, 0]),
            np.array([0, 1]),
            np.array([1, 0]),
            np.array([1, 1])
        ],
        expectedValues=[
            np.array([0]),
            np.array([1]),
            np.array([1]),
            np.array([1])
        ],
    )

    print(Neuron.predict(model, np.array([0, 0])))
    print(Neuron.predict(model, np.array([0, 1])))
    print(Neuron.predict(model, np.array([1, 0])))
    print(Neuron.predict(model, np.array([1, 1])))
