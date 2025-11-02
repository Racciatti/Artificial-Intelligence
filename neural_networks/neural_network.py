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
        
        [self.layers[i].activate(input) if i == 0 else self.layers[i].activate() for i in range(len(self.layers))]

    
    def calculate_loss(self, target_output) -> float:
        
        predictions = [neuron.activation for neuron in self.layers[-1].neurons]

        if len(predictions) != len(target_output):
            raise ValueError('The output layer number of neurons does not match the number of expected outputs')

        return (1/2) * sum([(target_output[i]-predictions[i])**2 for i in range(len(predictions))])
    
    
    
    def backpropagation(self, target_output:list, learning_rate:float):

        # Fetch last layer
        last_layer_neurons = self.layers[-1].neurons

        # Get the error signals
        predictions = [neuron.activation for neuron in last_layer_neurons]

        error_signals = list(map(lambda x, y:x-y, predictions, target_output))

        # Use the error signals and the chain rule to compute the weight change for the last layer

        # For each output neuron

        for i, neuron in enumerate(last_layer_neurons):

            # Each neuron has one bias, but can have multiple connections. 
            # Hence, in order to compute the bias gradient we need to take its average
            bias_gradients = []

            # For each connection into that output neuron
            for dendrite in neuron.dendrites:
                # DelC/Delw = delz/delw * dela/delz * delc/dela

                # DelC/Delw = a(l-1) * activation_funcion_derivative * error_signals
                weight_gradient = dendrite.start_neuron.activation * self.activation_function_derivative(dendrite.start_neuron.preactivation) * error_signals[i]
                print(f"old weight: {dendrite.weight}")
                print(f"weight gradient: {weight_gradient}")

                # In the case of bias, we may have multiple 
                # Delc/Delb = 1 * activation_funcion_derivative * error_signals
                bias_gradients.append(self.activation_function_derivative(dendrite.start_neuron.preactivation) * error_signals[i])

                # Delc/Dela = w(l) * activation_funcion_derivative * error_signals
                # ---

                # Update the weight, since for each connection to a last layer neuron there only needs to be one calculation
                dendrite.weight -= learning_rate * weight_gradient
                print(f"new weight: {dendrite.weight}")
            
            mean_bias_gradient = sum(bias_gradients)/len(bias_gradients)
            print(f"mean bias gradient: {mean_bias_gradient}")

            neuron.bias -= mean_bias_gradient


        # For each weight in the last layer

        # Use the chain rule sucessively based on the activation gradient
    

    def train(self, examples:list, targets:list, epochs:int, learning_rate:float, verbose:bool):

        if len(examples) != len(targets):
            raise ValueError('')
        

        for example, target in zip(examples,targets):

            self.feed(example)
            self.backpropagation([target], learning_rate = learning_rate)