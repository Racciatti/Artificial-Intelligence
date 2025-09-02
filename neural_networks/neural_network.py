from layer import Layer
class NeuralNetwork:

    def __init__(self, neurons_per_layer:list, activation_function):
        self.activation_function = activation_function
        self.neurons_per_layer = neurons_per_layer
        self.layers = []
        self.build()

    def create_layers(self):
        for neuron_count in self.neurons_per_layer:
            self.layers.append(Layer(self.activation_function,neuron_count))

    def connect_layers(self):
        layer_count = len(self.layers)
        for i in range(layer_count-1):
            self.layers[i].attach(self.layers[i+1])


    def build(self):
        self.create_layers()
        self.connect_layers()

    def feed(self, input):
        self.layers[0].activate(input)


