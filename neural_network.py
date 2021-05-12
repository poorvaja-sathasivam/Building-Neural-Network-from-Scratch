# Importing numpy
import np as np
import numpy as np

# X is the input
X = np.array(([0, 0, 0], [0, 0, 1], [0, 1, 0],
              [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]), dtype=float)

# y is the output for the neural network
y = np.array(([1], [0], [0], [0], [0],
              [0], [0], [1]), dtype=float)

# Values to be predicted
xPredicted = np.array(([0, 0, 1]), dtype=float)

# Maximum of X input array
X = X / np.amax(X, axis=0)

# Maximum of xPredicted (Input data for the Prediction)
xPredicted = xPredicted / np.amax(xPredicted, axis=0)

# Sets up the Loss file for graphing
lossFile = open("SumSquaredLossList.csv", "w")


# Neural network class
class NeuralNetwork(object):
    def __init__(self):
        # parameters
        self.inputLayerSize = 3  # X1,X2,X3
        self.outputLayerSize = 1  # Y1
        self.hiddenLayerSize = 4  # Size of the hidden layer
        # Building weights for each layer
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)  # 3X4 matrix for the hidden layer
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)  # 4x1 matrix for hidden layer to output

    def feedForward(self, X):
        # feedForward propagation through our network
        # dot product of X (input) and first set of 3x4 weights
        self.z = np.dot(X, self.W1)
        # the activationSigmoid activation function - neural magic
        self.z2 = activationSigmoid(self.z)
        # dot product of hidden layer (z2) and second set of 4x1 weights
        self.z3 = np.dot(self.z2, self.W2)
        # final activation function - more neural magic
        o = activationSigmoid(self.z3)
        return o

    def backwardPropagate(self, X, y, o):
        # backward propagate through the network
        # calculate the error in output
        self.o_error = y - o
        # apply derivative of activationSigmoid to error
        self.o_delta = self.o_error * activationSigmoidPrime(o)
        # z2 error: how much our hidden layer weights contributed to output
        # error
        self.z2_error = self.o_delta.dot(self.W2.T)
        # applying derivative of activationSigmoid to z2 error
        self.z2_delta = self.z2_error * activationSigmoidPrime(self.z2)
        # adjusting first set (inputLayer --&gt; hiddenLayer) weights
        self.W1 += X.T.dot(self.z2_delta)
        # adjusting second set (hiddenLayer --&gt; outputLayer) weights
        self.W2 += self.z2.T.dot(self.o_delta)

    def trainNetwork(self, X, y):
        # feed forward the loop
        o = self.feedForward(X)
        # and then back propagate the values (feedback)
        self.backwardPropagate(X, y, o)


def activationSigmoid(s):
    # activation function
    # simple activationSigmoid curve as in the book
    return 1 / (1 + np.exp(-s))


def activationSigmoidPrime(s):
    # First derivative of activationSigmoid
    # calculus time!
    return s * (1 - s)


def predictOutput(self):
    print("Predicted XOR output data based on trained weights: ")
    print("Expected (X1-X3): \n" + str(xPredicted))
    print("Output (Y1): \n" + str(self.feedForward(xPredicted)))


newNeuralNetwork = NeuralNetwork()
trainingEpochs = 1000
# trainingEpochs = 100000

for i in range(trainingEpochs):
    print("Epoch # " + str(i) + "\n")
    print("Network Input : \n" + str(X))
    print("Expected Output of XOR Gate Neural Network: \n" + str(y))
    print("Actual Output from XOR Gate Neural Network: \n" + str(newNeuralNetwork.feedForward(X)))
    # mean sum squared loss
    Loss = np.mean(np.square(y - newNeuralNetwork.feedForward(X)))
    newNeuralNetwork.saveSumSquaredLossList(i, Loss)
    print("Sum Squared Loss: \n" + str(Loss))
    print("\n")
    newNeuralNetwork.trainNetwork(X, y)

newNeuralNetwork.saveWeights()
newNeuralNetwork.predictOutput()
