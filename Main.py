from RedeNeural import RedeNeural
from TreinamentoRede import TreinamentoDaRede
from Grafico import Grafico

# Gabriel Nathan Almeida Silva, 14A
# Como executar, pode ser pelo python Jupter, copie e cole todos os arquivos ou 
# pip install matplotlib no terminal, certifique que tenha o Python 3 instalado

matrizDePeso = [
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 3, 0, 0, 0, 0],
    [0,-4, 1, 0, 0, 0],
    [0,-1,-3, 0, 0, 0],
    [0, 0, 0, 2,-10,0]
];

amostraDados = [
        (-3.0, 0.73212),
        (-2.0, 0.7339),
        (-1.0, 0.7838),
        (-0.5, 0.8903),
        (0.0,  0.982),
        (0.5,  0.8114),
        (1.0,  0.5937),
        (1.5,  0.5219),
        (2.0,  0.5049),
        (3.0,  0.5002)
];

rede = RedeNeural(matrizDePeso);
y = [rede.a1, rede.a2, rede.a3, rede.a4, rede.a5];
delta = [rede.delta1, rede.delta2, rede.delta3, rede.delta4];

treine = TreinamentoDaRede(matrizDePeso, y, delta, amostraDados);

print("Valores dos dados de teste x=0.0 e y = 0.0");
# Imprime os dados de teste para x=0.0 e yt=0.5
treine.imprimirDados(0.0, matrizDePeso, 0.5);
print("Valores dos dados de teste x=1.0 e y = 0.1");
# Imprime os dados de teste para x=1.0 e yt=0.1
treine.imprimirDados(1.0, matrizDePeso, 0.1);

plotGrafico = Grafico(treine);
plotGrafico.plotDoGrafico();
