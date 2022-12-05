import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

dados = pd.read_csv('dados_compra.csv')
modelo = LinearSVC()
seed = 20

# trocando os nomes das colunas para ptbr
mapa = {
  'home': 'principal',
  "how_it_works": 'como_funciona',
  'contact': 'contatos',
  'bought': 'comprou'
}
dados = dados.rename(columns = mapa)

# separando as colunas de input e output
x = dados[['principal', 'como_funciona', 'contatos']]
y = dados['comprou']

# Pegar os primeiros 75 elmentos para treinar a maquina 
# treino_x = x[:75]
# treino_y = y[:75]

# Pegar os ultimos elmentos para fazer os testes 
# teste_x = x[75:]
# teste_y = y[75:]

# essa é uma forma mais simples/melhorada e com menos codigo para replicar o codigo a cima

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, random_state = seed ,test_size = 0.25, stratify = y)

print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

# o metodo fit do sklearn é para a maquina treinar 
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

taxa_de_acerto = accuracy_score(teste_y, previsoes) * 100

print("Taxa de acerto: " + str(taxa_de_acerto.round()) + "%")

# teste_y.value_counts()
# treino_y.value_counts()