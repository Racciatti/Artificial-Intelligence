import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from neural_network import NeuralNetwork

logistic = lambda x: 1 / (1 + np.exp(-x))
logistic_derivative = lambda x: logistic(x) * (1 - logistic(x))

data = load_breast_cancer()
X = data.data
y = data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y_reshaped = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_reshaped, test_size=0.2, random_state=42
)

X_train_list = X_train.tolist()
y_train_list = y_train.tolist()
X_test_list = X_test.tolist()
y_test_list = y_test.tolist()

print(f"Number of features (inputs): {len(X_train_list[0])}")
print(f"Number of training examples: {len(X_train_list)}")

neuralNet = NeuralNetwork(
    neurons_per_layer=[30, 15, 5, 1], 
    activation_function=logistic,
    verbose=False,
    activation_function_derivative=logistic_derivative
)

print("\n--- STARTING TRAINING ---")
neuralNet.train(
    examples=X_train_list,
    targets=y_train_list,
    epochs=150,
    learning_rate=0.01,
    verbose=False
)
print("--- TRAINING COMPLETE ---")


correct_predictions = 0
for example, target in zip(X_test_list, y_test_list):

    prediction_raw = neuralNet.predict(example)[0]

    prediction_class = 1 if prediction_raw > 0.5 else 0

    if prediction_class == target[0]:
        correct_predictions += 1

accuracy = (correct_predictions / len(X_test_list)) * 100
print(f"\nAccuracy on Test Set: {accuracy:.2f}%")