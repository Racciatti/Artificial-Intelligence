from neurons import Neuron, Connection
from layer import Layer
from neural_network import NeuralNetwork

# Activation function
relu = lambda x : max(0,x)
identity = lambda x: x


neuralNet = NeuralNetwork([3,2,3],identity,verbose=True)

neuralNet.feed([i for i in range(3)])
