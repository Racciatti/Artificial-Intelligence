from neurons import Neuron, Connection
from layer import Layer
from neural_network import NeuralNetwork

# Activation functions
identity = lambda x: x
square = lambda x: x**2
leaky_relu = lambda x: x if x > 0 else 0.01 * x
der_leaky_relu = lambda x: 1 if x > 0 else 0.01

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

xor_examples = [ # LOSS: 4.785805690948886e-21
    [0,0],
    [0,1],
    [1,0],
    [1,1]
    ]
xor_targets = [[0], [1], [1], [0]]

neuralNet = NeuralNetwork(neurons_per_layer=[2,6,1], activation_function=leaky_relu,verbose=False, activation_function_derivative=der_leaky_relu)

neuralNet.train(examples=xor_examples, targets=xor_targets, epochs=50000, learning_rate=0.001, verbose=True)