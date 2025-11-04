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
    
    def backpropagation(self, target_output:list, learning_rate:float, verbose:bool = False):
        """
        Compute the gradient for the weights and biases using the backpropagation algorithm
        """


        # Fetch last layer
        last_layer_neurons = self.layers[-1].neurons

        # Get the error signals
        predictions = [neuron.activation for neuron in last_layer_neurons]
        print(predictions)
        print(target_output)
        error_signals = list(map(lambda x, y:x-y, predictions, target_output))

        # Use the error signals and the chain rule to compute the weight change for the last layer

        # For each output neuron
        for error_signal, neuron in zip(error_signals, last_layer_neurons):
            # For each connection into that output neuron
            
            
            for dendrite in neuron.dendrites:

                weight_gradient = dendrite.start_neuron.activation * self.activation_function_derivative(dendrite.start_neuron.preactivation) * error_signal
                if verbose: print(f"old weight: {dendrite.weight}")
                
                dendrite.weight -= learning_rate * weight_gradient
                if verbose: print(f"new weight: {dendrite.weight}")


            # Compute the bias gradient and update accordingly
            bias_gradient = self.activation_function_derivative(dendrite.start_neuron.preactivation) * error_signal
            if verbose: print(f"old bias: {neuron.bias}")
            neuron.bias -= learning_rate * bias_gradient
            if verbose: print(f"new bias: {neuron.bias}")
    

    def train(self, examples:list, targets:list, epochs:int, learning_rate:float, verbose:bool):

        if len(examples) != len(targets):
            raise ValueError('')
        
        for i in range(epochs):
            print(f"epoch {i}".upper())
            
            loss = []

            for example, target in zip(examples,targets):
                self.feed(example)
                self.backpropagation(target, learning_rate=learning_rate)

                loss.append(self.calculate_loss(target))
    
            print(f" LOSS: {sum(loss)/len(loss)}")

        print('Final weights:')
        neuron_dendrites = [neuron.dendrites for neuron in self.layers[-1].neurons]
        for i, neuron in enumerate(neuron_dendrites):
            print(f'neuron {i}')
            for j, dendrite in enumerate(neuron):
                print(f'dendrite {j}')
                print(dendrite.weight)
        



                
