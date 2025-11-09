import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlopen



# -----------------------------
# Função para calcular distância Euclidiana
# -----------------------------
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# -----------------------------
# Função KNN (classificação)
# -----------------------------
def knn_classify(train_df, test_point, k=5):
    distances = []
    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values
    
    for i, x in enumerate(X_train):
        dist = euclidean_distance(x, test_point)
        distances.append((dist, y_train[i]))
        
    # Ordenar pelas menores distâncias
    distances.sort(key=lambda x: x[0])
    
    # Pegar as k classes mais próximas
    neighbors = [cls for _, cls in distances[:k]]
    
    # Votação majoritária
    return max(set(neighbors), key=neighbors.count)




# -----------------------------
# Testar o modelo
# -----------------------------

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
cols = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
df = pd.read_csv(url, names=cols)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
train = df.iloc[:100]
test = df.iloc[100:]


X_test = test.iloc[:, :-1].values
y_test = test.iloc[:, -1].values

predictions = [knn_classify(train, x, k=5) for x in X_test]

accuracy = np.mean(np.array(predictions) == y_test)
print(f"Acurácia (KNN Classificador): {accuracy:.2f}")




# -----------------------------
# 6. Visualização simples
# -----------------------------
plt.figure(figsize=(8,6))
plt.scatter(df['petal_length'], df['petal_width'], c=pd.Categorical(df['species']).codes, cmap='viridis')
plt.title("Distribuição das espécies de Iris")
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.show()
