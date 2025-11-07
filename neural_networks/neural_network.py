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
        activation_gradients = []

        # From the last layer up to the second one
        for i, layer in enumerate(self.layers[:0:-1]):

            
            # If this is the last layer, we need to get the error signals
            if i == 0:

                # Get the error signals
                predictions = [neuron.activation for neuron in layer.neurons]
                error_signals = list(map(lambda x, y:x-y, predictions, target_output))

                # For each output neuron
                for error_signal, neuron in zip(error_signals, layer.neurons):
                    neuron_activation_gradients = []
                    
                    neuron_delta = error_signal * self.activation_function_derivative(neuron.preactivation)

                    # For each connection into that output neuron
                    for dendrite in neuron.dendrites:
                        
                        # Get its weight gradient and update it
                        weight_gradient = dendrite.start_neuron.activation * neuron_delta
                        dendrite.weight -= learning_rate * weight_gradient

                        # Get the connected neuron's activation gradient and store it
                        neuron_activation_gradients.append(dendrite.weight * neuron_delta)

                    # Store the gradients associated with the connections of a given neuron
                    activation_gradients.append(neuron_activation_gradients)

                    
                    bias_gradient = neuron_delta
                    neuron.bias -= learning_rate * bias_gradient

           # If this is not the last layer (i.e., a hidden layer)
            else:
                # Facilitate traversal
                current_activation_gradients_matrix = np.transpose(activation_gradients)
                activation_gradients = []

                for current_w_delta_terms, neuron in zip(current_activation_gradients_matrix, layer.neurons):
                    
                    # Sum of weighted errors (deltas) from the *next* layer
                    sum_weighted_error = sum(current_w_delta_terms) 
                    
                    # This neuron's delta (error signal)
                    neuron_error_signal = sum_weighted_error * self.activation_function_derivative(neuron.preactivation)
                    
                    new_activation_gradients_for_previous_layer = []
                    
                    # Update bias
                    bias_gradient = neuron_error_signal
                    neuron.bias -= learning_rate * bias_gradient

                    for dendrite in neuron.dendrites:
                        
                        # Get its weight gradient and update it
                        weight_gradient = dendrite.start_neuron.activation * neuron_error_signal
                        dendrite.weight -= learning_rate * weight_gradient

                        # Propagate the error term for the *next* iteration (the layer before this one)
                        new_activation_gradients_for_previous_layer.append(dendrite.weight * neuron_error_signal)

                    activation_gradients.append(new_activation_gradients_for_previous_layer)
                


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
        



                
