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
# Função KNN (regressão)
# -----------------------------
def knn_regress(train_df, test_point, k=5):
    distances = []
    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values
    
    for i, x in enumerate(X_train):
        dist = euclidean_distance(x, test_point)
        distances.append((dist, y_train[i]))
        
    distances.sort(key=lambda x: x[0])
    neighbors = [val for _, val in distances[:k]]
    
    # Média dos valores vizinhos
    return np.mean(neighbors)



# -----------------------------
# Testar modelo
# -----------------------------

url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df_reg = pd.read_csv(url)


df_reg = df_reg.sample(frac=1, random_state=42).reset_index(drop=True)
train = df_reg.iloc[:400]
test = df_reg.iloc[400:]


X_test = test.iloc[:, :-1].values
y_test = test.iloc[:, -1].values

predictions = [knn_regress(train, x, k=5) for x in X_test]

mae = np.mean(np.abs(predictions - y_test))
print(f"Erro médio absoluto (KNN Regressor): {mae:.2f}")



# -----------------------------
# 6. Visualização
# -----------------------------
plt.figure(figsize=(8,6))
plt.scatter(y_test, predictions, alpha=0.6)
plt.xlabel("Valor real")
plt.ylabel("Valor predito")
plt.title("KNN Regressor - Boston Housing")
plt.show()
