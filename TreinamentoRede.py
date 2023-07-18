import copy

from RedeNeural import RedeNeural

class TreinamentoDaRede:
    def __init__(self, matrizDePeso, y, delta, amostraDados):
        self.matrizDePeso = matrizDePeso;
        self.y = y;
        self.delta = delta;
        self.amostraDados = amostraDados;

    def gradientDescent(self, x, matrizDePeso, yt, rate=1):
        # Cria uma nova matriz como cópia profunda da matrizDePeso
        novaMatriz = copy.deepcopy(matrizDePeso);

        for i in range(len(matrizDePeso)):
            for j in range(len(matrizDePeso[i])):
                if novaMatriz[j][i] == 0:
                    continue;
                # Atualiza os pesos da novaMatriz usando a regra do Gradient Descent
                novaMatriz[j][i] = matrizDePeso[j][i] + rate * self.delta[i - 1](x, matrizDePeso, yt) * self.y[j - 1](x, matrizDePeso);
        return novaMatriz;

    def erros(self, x, matrizDePeso, yt):
        # Calcula o erro da rede para uma entrada x e valor alvo yt
        return (RedeNeural(matrizDePeso).a1(x, matrizDePeso) - yt) ** 2 / 2;
    

    def treinandoAmostra(self, n=10, novaMatriz=None):
        predicoes = [];
        todosErros = [];

        for i in range(n):
            predicao = [];
            erros = [];

            for i, j in self.amostraDados:
                # Define a matriz de pesos inicial como a matrizDePeso ou a novaMatriz se fornecida
                if novaMatriz is None:
                    novaMatriz = self.matrizDePeso;

                # Aplica o Gradient Descent para atualizar os pesos da novaMatriz
                novaMatriz = self.gradientDescent(i, novaMatriz, j);
                predicao.append((i, RedeNeural(self.matrizDePeso).a1(i, novaMatriz)));
                erros.append(self.erros(i, novaMatriz, j));

            predicoes.append(predicao);
            todosErros.append(erros);

        return novaMatriz, predicoes, todosErros;


    def imprimirPesos(self, matrizDePeso):
        print('matrizDePeso = [');
        for i in matrizDePeso:
            print(f'{i[1:]}');
        print(']');

    def imprimirDados(self, x, matrizDePeso, yt):
        print(f'x = {x:.2f}');
        print(f'y = {yt:.2f}');
        for i, o in enumerate(self.y):
            print(f'Z{i+1} = {o(x, matrizDePeso):.4f}');
        print();
        for i, j in enumerate(self.delta):
            print(f'Delta{i+1} = {j(x, matrizDePeso, yt):.4f}');
        print();
