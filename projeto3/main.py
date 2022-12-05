import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

dados = pd.read_csv('dados_sites.csv')
modelo = SVC()
seed = 20
np.random.seed(seed)
renomear = {
  'expected_hours': 'horas_esperadas',
  'price': 'preco',
  'unfinished': 'nao_finalizado'
}
dados = dados.rename(columns = renomear)
trocar = {
  1 : 0,
  0 : 1
}

dados['finalizado'] = dados.nao_finalizado.map(trocar)

x = dados[["horas_esperadas", "preco"]]
y = dados['finalizado']

raw_treino_x, raw_teste_x, treino_y, teste_y = train_test_split(x, y,test_size = 0.25, stratify = y)

scaler = StandardScaler()

scaler.fit(raw_treino_x)
treino_x = scaler.transform(raw_treino_x)
teste_x = scaler.transform(raw_teste_x)

print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)
taxa_de_acerto = accuracy_score(teste_y, previsoes) * 100

print("Taxa de acerto: " + str(taxa_de_acerto.round()) + "%")


# isso é uma linha de base para verificar se o acertos do chute estão satisfatorios 
# codigo para responder tudo sim
# baseLine = np.ones(540)
# taxa_de_acerto = accuracy_score(teste_y, baseLine) * 100
# print("Taxa de acerto da baseLine: " + str(taxa_de_acerto.round()) + "%")


# uma forma para testar se o algoritmo esta entendendo o que precisa ser feito 

# estou pegando os calores maximo e minimo das colunas e linhas

data_x = teste_x[:,0]
data_y = teste_x[:,1]

x_min = data_x.min()
x_max = data_x.max()
y_min = data_y.min()
y_max = data_y.max()

pixels = 100
eixo_x = np.arange(x_min, x_max, (x_max - x_min) / pixels)
eixo_y = np.arange(y_min, y_max, (y_max - y_min) / pixels)

# estou mesclando os pontos do eixo_x com o do eixo_y
xx, yy = np.meshgrid(eixo_x, eixo_y)
pontos = np.c_[xx.ravel(), yy.ravel()]


# testar todos os pixels das colonas e linhas do grafico
Z = modelo.predict(pontos)
# redimencionar os pontos para a tebela
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(data_x, data_y, c=teste_y, s=1)


# sns.scatterplot(x="horas_esperadas", y="preco", data=teste_x, hue=teste_y)


# fazer um grafico 
# sns.scatterplot(x="horas_esperadas", y="preco", data=dados, hue="finalizado")
# sns.relplot(x="horas_esperadas", y="preco", data=dados, hue="finalizado", col="finalizado")

plt.show()
