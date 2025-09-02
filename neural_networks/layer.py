from neurons import Neuron, Connection

class Layer:

    def __init__(self, activation_function, width:int):
        self.width = width
        self.neurons = [Neuron(activation_function=activation_function) for _ in range(self.width)]

    def attach(self, target_layer):
        for origin_neuron in self.neurons:
            for target_neuron in target_layer.neurons:
                print('connection created')
                connection = Connection(origin_neuron,target_neuron)
    
    def activate(self, input:list):
        if len(input) != len(self.neurons):
            raise ValueError(f'The input to this layer must have length {len(self.neurons)}!')
        
        for i in range(len(input)):
            self.neurons[i].activate(input[i])