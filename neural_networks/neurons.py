class Connection:

    def __init__(self, start_neuron:'Neuron', end_neuron:'Neuron', initial_weight:float = 0.5, verbose:bool = False):
        self.start_neuron = start_neuron
        self.end_neuron = end_neuron
        self.weight = initial_weight
        self.verbose = verbose

        self.activation = 0

        self.connect_neurons()

    def connect_neurons(self):
        self.start_neuron.add_axion(self)
        self.end_neuron.add_dendrite(self)
    
    def fire(self, value):
        if self.verbose: print('connection fired')
        self.activation = value * self.weight

class Neuron:

    def __init__(self, activation_function, initial_bias:float = 0, verbose:bool = False):
        self.activation_function = activation_function
        self.dendrites = []
        self.axions = []
        self.bias = initial_bias
        self.verbose = verbose
    
    def add_dendrite(self, connection:Connection):
        self.dendrites.append(connection)

    def add_axion(self, connection:Connection):
        self.axions.append(connection)

    def activate(self):

        activation = self.activation_function(sum(connection.activation for connection in self.dendrites) + self.bias)

        for axion in self.axions:
            axion.fire(value=activation)

        if self.verbose:(print(f'neuron activated with value {activation}'))