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

    
    def calculate_loss(self, target_output) -> float:
        
        predictions = [neuron.activation for neuron in self.layers[-1].neurons]

        if len(predictions) != len(target_output):
            raise ValueError('The output layer number of neurons does not match the number of expected outputs')

        return (1/2) * sum([(target_output[i]-predictions[i])**2 for i in range(len(predictions))])
    
    
    
    def backpropagation(self, target_output):

        # Get the error signals
        predictions = [neuron.activation for neuron in self.layers[-1].neurons]

        error_signals = predictions - target_output

        # Use the error signals and the chain rule to compute the weight change for the last layer

        # DelC/Delw = delz/delw * dela/delz * delc/dela

        # a(l-1) * activation_funcion_derivative * error_signals

        # For each weight in the last layer

        # Use the chain rule sucessively based on the activation gradient

