from layer import Layer
class NeuralNetwork:

    def __init__(self, neurons_per_layer:list, activation_function, activation_function_derivative, verbose:bool = False):
        self.verbose = verbose
        self.activation_function = activation_function
        self.neurons_per_layer = neurons_per_layer
        self.activation_function_derivative = activation_function_derivative
        self.layers = []
        self.build()

    def create_layers(self):
        for neuron_count in self.neurons_per_layer:
            self.layers.append(Layer(self.activation_function,neuron_count, verbose = self.verbose))

    def connect_layers(self):
        layer_count = len(self.layers)
        for i in range(layer_count-1):
            self.layers[i].attach(target_layer=self.layers[i+1])


    def build(self):
        self.create_layers()
        self.connect_layers()

    def feed(self, input:list):
        if self.verbose: print('net fed')
        
        [self.layers[i].activate(input) if i == 0 else self.layers[i].activate() for i in range(len(self.layers))]


