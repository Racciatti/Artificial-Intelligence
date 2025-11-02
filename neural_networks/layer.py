from neurons import Neuron, Connection

class Layer:

    def __init__(self, activation_function, width:int, verbose:bool = False):
        self.verbose = verbose
        self.width = width
        self.neurons = [Neuron(activation_function=activation_function, verbose=self.verbose) for _ in range(self.width)]

    def attach(self, target_layer):
        for origin_neuron in self.neurons:
            for target_neuron in target_layer.neurons:
                connection = Connection(origin_neuron,target_neuron)
    
    
    def activate(self, input:list= []):

        # IF THIS IS AN INPUT LAYER
        if input != []:
            for i, neuron in enumerate(self.neurons):
                neuron.activation = input[i]
                for axion in neuron.axions:
                    axion.fire(neuron.activation)
        else:
            for neuron in self.neurons: neuron.activate()