from neurons import Neuron, Connection
from layer import Layer
from neural_network import NeuralNetwork

# Activation functions
relu = lambda x : max(0,x)
identity = lambda x: x
square = lambda x: x**2
der_relu = lambda x : 1 if x > 0 else 0

# Fit tests
or_examples = [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
    ]
or_targets = [0, 1, 1, 1]

xor_examples = [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
    ]
xor_targets = [0, 1, 1, 0]


neuralNet = NeuralNetwork(neurons_per_layer=[2,1], activation_function=identity,verbose=False, activation_function_derivative=lambda x : 1)

neuralNet.train(or_examples, or_targets, 100, 0.01, True)