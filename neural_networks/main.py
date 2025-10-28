from neurons import Neuron, Connection
from layer import Layer
from neural_network import NeuralNetwork

# Activation function
relu = lambda x : max(0,x)
identity = lambda x: x
square = lambda x: x**2

der_relu = lambda x : 1 if x > 0 else 0


xor_examples = [
    [0,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,0]
    ]

neuralNet = NeuralNetwork(neurons_per_layer=[3,2,1], activation_function=identity,verbose=True)

neuralNet.feed([i for i in range(3)])
