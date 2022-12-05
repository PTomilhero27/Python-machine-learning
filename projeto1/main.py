from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from testes import *

model = LinearSVC()
model.fit(treino_x(), trieno_y())

misterio1 = [1,1,1]
misterio2 = [1,1,0]
misterio3 = [0,1,1]

testes = [misterio1,misterio2,misterio3]
previsoes = model.predict(testes)

testes_classes = [0,1,1]

# metodo para pegar a % de acertos nas previsoes
# corretos = (previsoes == testes_classes).sum()
# total = len(testes)
# taxa_de_acerto = (corretos/total) * 100
# print("Taxa de acerto: " + str(taxa_de_acerto.round()) + "%")

# função do sklearn para pegar a % de acertos de forma mais simples
# primeiro parametro é os valores verdadeiros e depois as previsoes 
taxa_de_acerto = accuracy_score(testes_classes, previsoes) * 100

print("Taxa de acerto: " + str(taxa_de_acerto.round()) + "%")

# mesmo resultados com menos linha de codigo 