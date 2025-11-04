from layer import Layer
import numpy as np

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

        activation_gradients_per_neuron = []
        # From the last layer up to the second one
        for i, layer in enumerate(self.layers[:0:-1]):

            
            # If this is the last layer, we need to get the error signals
            if i == 0:
                print('CALLED')

                # Get the error signals
                predictions = [neuron.activation for neuron in layer.neurons]
                error_signals = list(map(lambda x, y:x-y, predictions, target_output))

                # For each output neuron
                for error_signal, neuron in zip(error_signals, layer.neurons):
                    activation_gradients = []
                    
                    # For each connection into that output neuron
                    for dendrite in neuron.dendrites:
                        
                        # Get its weight gradient and update it
                        weight_gradient = dendrite.start_neuron.activation * self.activation_function_derivative(dendrite.start_neuron.preactivation) * error_signal
                        if verbose: print(f"old weight: {dendrite.weight}")
                        dendrite.weight -= learning_rate * weight_gradient
                        if verbose: print(f"new weight: {dendrite.weight}")

                        # Get the connected neuron's activation gradient and store it
                        activation_gradients.append(dendrite.weight * self.activation_function_derivative(dendrite.start_neuron.preactivation) * error_signal)

                    # Store the gradients associated with the connections of a given neuron
                    activation_gradients_per_neuron.append(activation_gradients)

                    # Get the bias' gradient and update it
                    bias_gradient = self.activation_function_derivative(dendrite.start_neuron.preactivation) * error_signal
                    if verbose: print(f"old bias: {neuron.bias}")
                    neuron.bias -= learning_rate * bias_gradient
                    if verbose: print(f"new bias: {neuron.bias}")

            # If this is not the last layer
            else:
                # Manipulate the stored activation's so they have a better format for traversal (transposing)
                print(activation_gradients_per_neuron)
                current_activation_gradients = np.transpose(activation_gradients_per_neuron)
                print(current_activation_gradients)
                
                # For each neuron in the current layer 

                # Compute the gradient updates

                # Store the next layer's neurons' activation gradients
                pass
                


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
        



                
