from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

class Grafico:
    def __init__(self, amostraDados):
        self.amostraDados = amostraDados

    def plotDoGrafico(self):
        # Obtendo os dados da amostra, predições e erros
        matriz, predicoes, todosErros = self.amostraDados.treinandoAmostra(1000);
        erros = [];

        # Combinando todos os erros em uma única lista
        for i in todosErros:
            erros.extend(i);

        # Imprimindo a matriz de pesos
        self.amostraDados.imprimirPesos(matriz);
        
        # Criando a figura e os subplots
        imagem, plotar = plt.subplots(nrows=2);
        ln, = plotar[1].plot([], []);
        
        # Definindo os limites dos eixos
        plotar[1].set_xlim(-10, 10);
        plotar[1].set_ylim(0, 2);
        
        def atualizar(frame):
            # Função de atualização do gráfico a cada frame
            
            dadosX = [];
            dadosY = [];

            # Obtendo os dados de x e y para o frame atual
            for i, j in predicoes[frame]:
                dadosX.append(i);
                dadosY.append(j);

            ln.set_data(dadosX, dadosY);

            return ln,

        # Plotando os erros
        plotar[0].plot(erros);
        
        # Criando a animação do gráfico
        linhaGrafico = FuncAnimation(imagem, atualizar, frames=len(predicoes), interval=20, blit=True);
        
        # Exibindo o gráfico
        plt.show();
