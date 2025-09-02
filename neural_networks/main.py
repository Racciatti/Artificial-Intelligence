from neurons import Neuron, Connection
from layer import Layer
from neural_network import NeuralNetwork

# Activation function
relu = lambda x : max(0,x)

neuralNet = NeuralNetwork([10,20,10],relu)
neuralNet.feed([i for i in range(10)])