import numpy as np

#Implementação da regressão logística
class regLogs:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.peso = None #theta
        self.bias = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def fit(self, X, y):
        interception = np.c_[np.ones((X.shape[0], 1)), X]  # Adiciona coluna de 1s para o bias
        n_samples, n_features = interception.shape # Agora inclui o bias como uma feature
        self.peso = np.zeros(n_features)

        for _ in range(self.n_iters):
            predictions = self.sigmoid(interception.dot(self.peso))
            errors = predictions - y
            gradient = interception.T.dot(errors) / n_samples
            self.peso -= self.learning_rate * gradient

        self.bias = self.peso[0]  
        self.peso = self.peso[1:]      

    def probabilidade(self, X):
        interception = np.c_[np.ones((X.shape[0], 1)), X]
        return self.sigmoid(interception.dot(np.r_[self.bias, self.peso])) #Calculuca a probabilidade usando o bias
    
    def predicao(self, X, limiar=0.5):
        probabilidades = self.probabilidade(X)
        return (probabilidades >= limiar).astype(int) #Retorna 1 se a probabilidade for maior ou igual ao limiar, caso contrário retorna 0