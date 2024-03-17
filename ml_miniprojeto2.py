#Importando bibliotecas
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

#Carregando o conjunto de dados
dados = pd.read_csv("dados/dataset.csv")

print(dados.shape)
print(dados.info())
print(dados.sample(10))

#Análise Exploratória
print(dados.corr())
sns.pairplot(dados)
plt.show()

#Pré-Processamento
X = dados[['tempo_cadastro_cliente', 'numero_medio_cliques_por_sessao', 'tempo_total_logado_app', 'tempo_total_logado_website']]
y = dados['valor_total_gasto']

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.3, random_state = 101)
print(len(X_treino))
print(len(X_teste))

#Padronização dos dados
scaler = StandardScaler()
scaler.fit(X_treino)
X_treino = scaler.transform(X_treino)
X_teste = scaler.transform(X_teste)