"""
Main para testar as implementações de Árvore de Decisão
- Classificação (usando Gini)
- Regressão (usando MSE)
"""

import pandas as pd
import numpy as np
from decisiontree_class import ArvoreDecisaoClassificador
from decisiontree_reg import ArvoreDecisaoRegressor


# ============================================================================
# MÉTRICAS DE AVALIAÇÃO
# ============================================================================

def calcular_accuracy(y_true, y_pred):
    """Calcula a acurácia"""
    return np.mean(y_true == y_pred)


def calcular_precision(y_true, y_pred, classe_positiva=1):
    """Calcula a precisão para uma classe específica"""
    tp = np.sum((y_true == classe_positiva) & (y_pred == classe_positiva))
    fp = np.sum((y_true != classe_positiva) & (y_pred == classe_positiva))
    
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def calcular_recall(y_true, y_pred, classe_positiva=1):
    """Calcula o recall para uma classe específica"""
    tp = np.sum((y_true == classe_positiva) & (y_pred == classe_positiva))
    fn = np.sum((y_true == classe_positiva) & (y_pred != classe_positiva))
    
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def calcular_f1_score(y_true, y_pred, classe_positiva=1):
    """Calcula o F1-Score"""
    precision = calcular_precision(y_true, y_pred, classe_positiva)
    recall = calcular_recall(y_true, y_pred, classe_positiva)
    
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def calcular_mse(y_true, y_pred):
    """Calcula o Mean Squared Error"""
    return np.mean((y_true - y_pred) ** 2)


def calcular_rmse(y_true, y_pred):
    """Calcula o Root Mean Squared Error"""
    return np.sqrt(calcular_mse(y_true, y_pred))


def calcular_mae(y_true, y_pred):
    """Calcula o Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))


def calcular_r2(y_true, y_pred):
    """Calcula o R² (coeficiente de determinação)"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)


def imprimir_metricas_classificacao(y_true, y_pred, nome_dataset="Dataset"):
    """Imprime todas as métricas de classificação"""
    print(f"\n{'='*60}")
    print(f"MÉTRICAS - {nome_dataset}")
    print(f"{'='*60}")
    
    accuracy = calcular_accuracy(y_true, y_pred)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Para cada classe
    classes = np.unique(y_true)
    print(f"\nMétricas por classe:")
    for classe in classes:
        precision = calcular_precision(y_true, y_pred, classe)
        recall = calcular_recall(y_true, y_pred, classe)
        f1 = calcular_f1_score(y_true, y_pred, classe)
        print(f"  Classe {classe}:")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall:    {recall:.4f}")
        print(f"    F1-Score:  {f1:.4f}")


def imprimir_metricas_regressao(y_true, y_pred, nome_dataset="Dataset"):
    """Imprime todas as métricas de regressão"""
    print(f"\n{'='*60}")
    print(f"MÉTRICAS - {nome_dataset}")
    print(f"{'='*60}")
    
    mse = calcular_mse(y_true, y_pred)
    rmse = calcular_rmse(y_true, y_pred)
    mae = calcular_mae(y_true, y_pred)
    r2 = calcular_r2(y_true, y_pred)
    
    print(f"MSE:   {mse:.4f}")
    print(f"RMSE:  {rmse:.4f}")
    print(f"MAE:   {mae:.4f}")
    print(f"R²:    {r2:.4f}")


# ============================================================================
# TESTE 1: CLASSIFICAÇÃO - Marketing Campaign Dataset
# ============================================================================

def teste_marketing_campaign():
    print("\n" + "="*70)
    print("TESTE 1: CLASSIFICAÇÃO - Marketing Campaign Dataset")
    print("="*70)
    
    # Carregar dataset
    df = pd.read_csv('data/marketing_campaign.csv', sep='\t')
    print(f"Dataset carregado: {df.shape[0]} amostras, {df.shape[1]} colunas")
    
    # Preparar dados - vamos prever 'Response' (se aceita a campanha)
    # Selecionar apenas colunas numéricas relevantes
    features = ['Year_Birth', 'Income', 'Recency', 'MntWines', 'MntFruits',
                'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
                'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
                'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']
    
    target = 'Response'
    
    # Remover NaN
    df_clean = df[features + [target]].dropna()
    
    X = df_clean[features].values
    y = df_clean[target].values
    
    print(f"Dados limpos: {len(y)} amostras")
    print(f"Classes: {np.unique(y)} (distribuição: {np.bincount(y)})")
    
    # Dividir em treino e teste (80/20)
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    
    train_idx, test_idx = indices[:split], indices[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"Treino: {len(y_train)} | Teste: {len(y_test)}")
    
    # Treinar árvore
    arvore = ArvoreDecisaoClassificador(
        profundidade_max=6,
        min_amostras_split=10,
        min_gini_melhoria=0.001
    )
    
    print("Treinando árvore...")
    arvore.fit(X_train, y_train)
    print("✓ Árvore treinada!")
    
    # Predições
    y_pred_train = arvore.predict(X_train)
    y_pred_test = arvore.predict(X_test)
    
    # Métricas
    imprimir_metricas_classificacao(y_train, y_pred_train, "TREINO")
    imprimir_metricas_classificacao(y_test, y_pred_test, "TESTE")


# TESTE 2: REGRESSÃO - Retail Store Inventory Dataset

def teste_retail_store():
    print("\n" + "="*70)
    print("TESTE 2: REGRESSÃO - Retail Store Inventory Dataset")
    print("="*70)
    
    # Carregar dataset
    df = pd.read_csv('data/retail_store_inventory.csv')
    print(f"Dataset carregado: {df.shape[0]} amostras, {df.shape[1]} colunas")
    
    # Preparar dados - vamos prever 'Units Sold'
    # Selecionar apenas colunas numéricas
    features = ['Inventory Level', 'Units Ordered', 'Demand Forecast', 
                'Price', 'Discount', 'Holiday/Promotion', 'Competitor Pricing']
    
    target = 'Units Sold'
    
    # Remover NaN
    df_clean = df[features + [target]].dropna()
    
    X = df_clean[features].values
    y = df_clean[target].values.astype(float)
    
    print(f"Dados limpos: {len(y)} amostras")
    print(f"Range do target: [{y.min():.2f}, {y.max():.2f}]")
    print(f"Média do target: {y.mean():.2f}")
    
    # Amostrar para acelerar o treinamento (usar apenas 5000 amostras)
    if len(X) > 5000:
        np.random.seed(42)
        indices = np.random.choice(len(X), 5000, replace=False)
        X = X[indices]
        y = y[indices]
        print(f"Amostragem: usando {len(y)} amostras para treino")
    
    # Dividir em treino e teste (80/20)
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    
    train_idx, test_idx = indices[:split], indices[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"Treino: {len(y_train)} | Teste: {len(y_test)}")
    
    # Treinar árvore
    arvore = ArvoreDecisaoRegressor(
        profundidade_max=8,
        min_amostras_split=20,
        min_mse_melhoria=0.5
    )
    
    print("Treinando árvore...")
    arvore.fit(X_train, y_train)
    print("✓ Árvore treinada!")
    
    # Predições
    y_pred_train = arvore.predict(X_train)
    y_pred_test = arvore.predict(X_test)
    
    # Métricas
    imprimir_metricas_regressao(y_train, y_pred_train, "TREINO")
    imprimir_metricas_regressao(y_test, y_pred_test, "TESTE")



if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTANDO IMPLEMENTAÇÕES DE ÁRVORE DE DECISÃO - DATASETS REAIS")
    print("="*70)
    
    # Executar testes com datasets reais
    teste_marketing_campaign()
    teste_retail_store()

