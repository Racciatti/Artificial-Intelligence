class Connection:

    def __init__(self, start_neuron:'Neuron', end_neuron:'Neuron', initial_weight:float = 0):
        self.start_neuron = start_neuron
        self.end_neuron = end_neuron
        self.weight = initial_weight

        self.connect_neurons()

    def connect_neurons(self):
        self.start_neuron.add_axion(self)
        self.end_neuron.add_dendrite(self)
    
    def fire(self, value):
        print('connection fired')
        self.end_neuron.activate(value * self.weight)

class Neuron:

    def __init__(self, activation_function):
        self.activation_function = activation_function
        self.dendrites = []
        self.axions = []
    
    def add_dendrite(self, connection:Connection):
        self.dendrites.append(connection)

    def add_axion(self, connection:Connection):
        self.axions.append(connection)

    def activate(self, value:float):
        print('neuron activated')
        activation = self.activation_function(value)
        for axion in self.axions:
            axion.fire(value=activation)