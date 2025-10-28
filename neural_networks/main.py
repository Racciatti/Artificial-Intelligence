from neurons import Neuron, Connection
from layer import Layer
from neural_network import NeuralNetwork

# Activation function
relu = lambda x : max(0,x)

neuralNet = NeuralNetwork([2,2],relu,verbose=True)
neuralNet.feed([i for i in range(2)])
