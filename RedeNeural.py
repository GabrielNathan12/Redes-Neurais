import math

class RedeNeural:
    
    def __init__(self, matrizDePeso):
        self.matrizDePeso = matrizDePeso;

    def sigmoidal(self, x):
        return 1 / (1 + math.exp(-x));
    
    def derivadaSigmoidal(self, x):
        return x * (1 - x);
    
    def relux(self, x):
        if 0 > x:
            return 0;
        else:
            return x;

    def derivadaRelux(self, x):
        if 0 >= x:
            return 0;
        else:
            return 1;

    def a5(self, x, matrizDePeso):
        return x;

    def a4(self, x, matrizDePeso):
        # Cálculo da saída do neurônio a4
        return self.sigmoidal(matrizDePeso[5][4] * self.a5(x, matrizDePeso));

    def a3(self, x, matrizDePeso):
        # Cálculo da saída do neurônio a3
        return self.sigmoidal(matrizDePeso[5][3] * self.a5(x, matrizDePeso));

    def a2(self, x, matrizDePeso): 
        # Cálculo da saída do neurônio a2
        return self.relux(matrizDePeso[3][2] * self.a3(x, matrizDePeso) + matrizDePeso[4][2] * self.a4(x, matrizDePeso));

    def a1(self, x, matrizDePeso):
        # Cálculo da saída do neurônio a1
        return self.sigmoidal(matrizDePeso[3][1] * self.a3(x, matrizDePeso) + matrizDePeso[2][1] * self.a2(x, matrizDePeso)
                            + matrizDePeso[4][1] * self.a4(x, matrizDePeso));

    def delta1(self, x, matrizDePeso, yt):
        y = self.a1(x, matrizDePeso);
        # Cálculo do delta do neurônio a1
        return self.derivadaSigmoidal(y) * (yt - y);

    def delta2(self, x, matrizDePeso, yt):
        y = self.a2(x, matrizDePeso);
        # Cálculo do delta do neurônio a2
        return self.derivadaRelux(y) * matrizDePeso[2][1] * self.delta1(x, matrizDePeso, yt);

    def delta3(self, x, matrizDePeso, yt):
        y = self.a3(x, matrizDePeso);
        # Cálculo do delta do neurônio a3
        resultado = matrizDePeso[3][2] * self.delta2(x, matrizDePeso, yt) + matrizDePeso[3][1] * self.delta1(x, matrizDePeso, yt);
        return self.derivadaSigmoidal(y) * resultado;

    def delta4(self, x, matrizDePeso, yt):
        y = self.a4(x, matrizDePeso);
        # Cálculo do delta do neurônio a4
        resultado = matrizDePeso[4][2] * self.delta2(x, matrizDePeso, yt) + matrizDePeso[4][1] * self.delta1(x, matrizDePeso, yt);
        return self.derivadaSigmoidal(y) * resultado;
