# 🐍 Snake Q-Learning AI

Um projeto de **Inteligência Artificial** que ensina um agente a jogar Snake usando **Q-Learning**, uma técnica de Reinforcement Learning.

## 🎯 Sobre o Projeto

Este projeto implementa um agente inteligente que aprende a jogar Snake através de Q-Learning.

### 🎥 Inspiração

Baseado no vídeo: [Q-Learning Snake AI](https://youtu.be/je0DdS0oIZk)

## 🚀 Como Executar

### Pré-requisitos

```bash
pip install -r requirements.txt
```

### Executar o Projeto

```bash
python snakeql.py
```

### Opções Disponíveis

1. **Treinar do Zero** - Inicia um novo treinamento
2. **Testar Modelo** - Visualiza um modelo já treinado jogando
3. **Continuar Treinamento** - Continua o treinamento de um modelo existente

## 📁 Estrutura do Projeto

```
📦 snake-q-learning
├── 🧠 snakeql.py          # Algoritmo Q-Learning principal
├── 🎮 snake_no_visual.py  # Ambiente do jogo (sem gráficos)
├── 🎨 visualsnake.py      # Visualização com Pygame
├── 📦 pickle/             # Modelos treinados salvos
├── 📋 requirements.txt    # Dependências
└── 📖 README.md          # Este arquivo
```

### 🔍 Detalhamento dos Arquivos

#### 🧠 `snakeql.py` - O Cérebro (Q-Learning)

- ✅ Classe `SnakeQ()` com a **tabela Q**
- ✅ Função `get_action()` - escolhe ações (exploração vs exploitação)
- ✅ Função `train()` - **algoritmo de Q-Learning**
- ✅ **Equação de Bellman** para atualizar a tabela Q
- ✅ Sistema de salvamento automático de modelos

#### 🎮 `snake_no_visual.py` - Ambiente do Jogo

- ✅ Lógica completa do Snake (movimento, colisões, comida)
- ✅ Função `get_state()` - retorna o estado atual (12 features)
- ✅ Função `step(action)` - executa uma ação e retorna (estado, recompensa, game_over)
- ⚡ **Otimizado para velocidade** - sem gráficos para treinamento rápido

#### 🎨 `visualsnake.py` - Visualização

- ✅ Interface gráfica com **Pygame**
- ✅ Função `run_game()` - carrega modelo treinado e mostra jogando
- ✅ Múltiplas partidas automáticas
- ✅ Estatísticas em tempo real

### Sistema de Recompensas

| Situação                | Recompensa | Objetivo                |
| ----------------------- | ---------- | ----------------------- |
| 🍎 **Comeu comida**     | **+1**     | Incentiva crescimento   |
| 👟 **Movimento normal** | **0**      | Neutro (sem penalidade) |
| 💀 **Morreu**           | **-10**    | Evita colisões          |

### Parâmetros de Treinamento

```python
learning_rate = 0.1      # Taxa de aprendizado
discount_rate = 0.95     # Fator de desconto
eps = 1.0               # Exploração inicial
eps_discount = 0.9995   # Decaimento da exploração
```

## 🛠️ Tecnologias Utilizadas

- **Python 3.8+**
- **NumPy** - Cálculos numéricos e manipulação de arrays
- **Pygame** - Interface gráfica e visualização
- **Pickle** - Serialização e salvamento de modelos

## 🎮 Controles

Durante a visualização:

- **ESC** ou **fechar janela** - Parar a execução
- **Ctrl+C** no terminal - Interromper treinamento

## 👨‍💻 Autor

**Nicolas** - [GitHub](https://github.com/sleeper02)

---
