from datetime import datetime

import graphviz
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz

dados = pd.read_csv('dados_carros.csv')
seed = 20
np.random.seed(seed)
dummy = DummyClassifier()
# modelo = SVC()
modelo = DecisionTreeClassifier(max_depth=2)

renomear = {
  "mileage_per_year" : 'milhas_por_ano',
  "model_year": "ano_do_modelo",
  "price": "preco",
  "sold": "vendido",
}

trocar = {
  "no" : 0,
  "yes": 1
}



dados = dados.rename(columns=renomear)
dados.vendido = dados.vendido.map(trocar)

# estou criando uma coluna para classificar a idade do carro para melhorar a classificação do codigo 
ano_atual = datetime.today().year
dados['idade_do_modelo'] = ano_atual - dados.ano_do_modelo 

# acresentando mais uma coluna para km no lugar de milhas
dados['km_por_ano'] = dados.milhas_por_ano * 1.60934

# vamos excluir as colunas que não vamos ultilizar 
dados = dados.drop(columns=['Unnamed: 0', "milhas_por_ano", "ano_do_modelo"], axis=1)

# pegar os inputs X e os outputs Y
x = dados[['preco', "idade_do_modelo", "km_por_ano"]]
y = dados['vendido']

# treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.25, stratify=y)

# print("treinamos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

# modelo.fit(treino_x, treino_y)
# previsoes = modelo.predict(teste_x)

# texa_de_acerto = accuracy_score(teste_y, previsoes) * 100
# print("A taxa de acerto foi %.0f%%" % texa_de_acerto.round())

# dummy.fit(treino_x, treino_y)
# dummy_previsoes = dummy.predict(teste_x)

# texa_de_acerto = accuracy_score(teste_y, dummy_previsoes) * 100

# print("A taxa de acerto foi %.0f%%" % texa_de_acerto.round())


# esse é um grupo de codigo para redimencionar as escalas 
# raw_treino_x, raw_teste_x, treino_y, teste_y = train_test_split(x, y,test_size = 0.25, stratify = y)


# scaler = StandardScaler()

# scaler.fit(raw_treino_x)
# treino_x = scaler.transform(raw_treino_x)
# teste_x = scaler.transform(raw_teste_x)

# print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

# modelo.fit(treino_x, treino_y)
# previsoes = modelo.predict(teste_x)
# taxa_de_acerto = accuracy_score(teste_y, previsoes) * 100

# print("Taxa de acerto: " + str(taxa_de_acerto.round()) + "%")

# criando um algoritmo com uma arvore 


raw_treino_x, raw_teste_x, treino_y, teste_y = train_test_split(x, y,test_size = 0.25, stratify = y)


# scaler = StandardScaler()

# scaler.fit(raw_treino_x)
# treino_x = scaler.transform(raw_treino_x)
# teste_x = scaler.transform(raw_teste_x)

print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(raw_treino_x), len(raw_teste_x)))

modelo.fit(raw_treino_x, treino_y)
previsoes = modelo.predict(raw_treino_x)
taxa_de_acerto = accuracy_score(teste_y, previsoes) * 100

print("Taxa de acerto: " + str(taxa_de_acerto.round()) + "%")

features = x.columns
dot_data = export_graphviz(modelo, out_file=None, feature_names=features, filled=True, rounded=True,
                          class_names= ['não', 'sim'])
grafico = graphviz.Source(dot_data)
grafico.view()