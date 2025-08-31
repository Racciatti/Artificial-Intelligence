from abc import ABC, abstractmethod

class AbstractRunner(ABC):

    def __init__(self):
        pass
    
    @abstractmethod
    def log(self):
    # Logar as informações de forma estruturada em um arquivo, para uso posterior de visualização
        pass

    @abstractmethod
    def validate(self):
    # Receber a q table atual e rodar testes com ela para extrair e retornar certas métricas:
    # Mudança nos valores tabulares, número de passos para passar nos testes (max, min, médio)
        pass

    @abstractmethod
    def run(self):
    # Rodar o q learning com base nos hiperparâmetros passados, obtendo certas métricas a cada x épocas de forma que se possa visualizar
    # como o aprendizado evolui ao longo do tempo. As métricas obtidas são logadas em um arquivo csv (nome depende dos parâmetros)
    # ao final da execução do algoritmo
        pass

    @abstractmethod
    def view(self):
    # Com base em um log estruturado, gerar visualizações 
        pass

class MazeRunner(AbstractRunner):
    pass

class TaxiRunner(AbstractRunner):
    pass

mazeRunner = MazeRunner()