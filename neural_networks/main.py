from neurons import Neuron, Connection
from layer import Layer
from neural_network import NeuralNetwork

# Activation functions
relu = lambda x : max(0,x)
identity = lambda x: x
square = lambda x: x**2
der_relu = lambda x : 1 if x > 0 else 0

# TESTS
# 2,1 net
or_examples = [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
    ]
or_targets = [0, 1, 1, 1]

# 2,3 net
extended_or_examples = [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
    ]
extended_or_targets = [
    [1,1,1],
    [1,0,1],
    [0,1,1],
    [0,0,1]
    ]

xor_examples = [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
    ]
xor_targets = [[0], [1], [1], [0]]


neuralNet = NeuralNetwork(neurons_per_layer=[2,2,1], activation_function=identity,verbose=False, activation_function_derivative=lambda x: 1)

neuralNet.train(examples=xor_examples, targets=xor_targets, epochs=10, learning_rate=0.001, verbose=True)