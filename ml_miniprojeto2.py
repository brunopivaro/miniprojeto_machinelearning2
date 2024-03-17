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

#Criando os modelos de regressão
modelo_v1 = LinearRegression()
modelo_v1.fit(X_treino, y_treino)
print("Coeficientes \n", modelo_v1.coef_)

#Salvando os coeficientes
df_coef = pd.DataFrame(modelo_v1.coef_, X.columns, columns = ['Coeficiente'])
print(df_coef.head(10))

#Avaliando o modelo
pred_v1 = modelo_v1.predict(X_teste)
plt.scatter(x = y_teste, y = pred_v1)
plt.xlabel('Valor real de Y')
plt.ylabel('Valor previsto de Y')
plt.show()

#Calculando o erro médio absoluto
print(mean_absolute_error(y_teste, pred_v1))
print(dados['valor_total_gasto'].mean()) #Comparando com a média da variável target

#Erro quadrático médio
print(np.sqrt(mean_squared_error(y_teste, pred_v1)))

#Coeficiente R2
print(r2_score(y_teste, pred_v1)) #Mais perto de 1 melhor

#Variância Explicada
print(explained_variance_score(y_teste, pred_v1))

#Regressão Ridge
modelo_v2 = Ridge(alpha = 1.0)
modelo_v2.fit(X_treino, y_treino)
print("Coeficientes \n", modelo_v2.coef_)

#Salvando os coeficientes
df_coef = pd.DataFrame(modelo_v2.coef_, X.columns, columns = ['Coeficiente'])
print(df_coef.head(10))

#Avaliando o modelo
pred_v2 = modelo_v2.predict(X_teste)
plt.scatter(x = y_teste, y = pred_v2)
plt.xlabel('Valor real de Y')
plt.ylabel('Valor previsto de Y')
plt.show()

#Calculando o erro médio absoluto
print(mean_absolute_error(y_teste, pred_v2))
print(dados['valor_total_gasto'].mean()) #Comparando com a média da variável target

#Erro quadrático médio
print(np.sqrt(mean_squared_error(y_teste, pred_v2)))

#Coeficiente R2
print(r2_score(y_teste, pred_v2)) #Mais perto de 1 melhor

#Variância Explicada
print(explained_variance_score(y_teste, pred_v2))

#Regressão Lasso
modelo_v3 = Lasso(alpha = 1.0)
modelo_v3.fit(X_treino, y_treino)
print("Coeficientes \n", modelo_v3.coef_)

#Salvando os coeficientes
df_coef = pd.DataFrame(modelo_v3.coef_, X.columns, columns = ['Coeficiente'])
print(df_coef.head(10))

#Avaliando o modelo
pred_v3 = modelo_v3.predict(X_teste)
plt.scatter(x = y_teste, y = pred_v3)
plt.xlabel('Valor real de Y')
plt.ylabel('Valor previsto de Y')
plt.show()

#Calculando o erro médio absoluto
print(mean_absolute_error(y_teste, pred_v3))
print(dados['valor_total_gasto'].mean()) #Comparando com a média da variável target

#Erro quadrático médio
print(np.sqrt(mean_squared_error(y_teste, pred_v3)))

#Coeficiente R2
print(r2_score(y_teste, pred_v3)) #Mais perto de 1 melhor

#Variância Explicada
print(explained_variance_score(y_teste, pred_v3))

#Seleção do modelo
#O modelo 3 apresentou uma leve taxa maior de erros, portanto iremos continuar apenas com o modelo 1 o e modelo 2
#Ambos os modelos trouxeram resultados muito próximos, portanto devemos escolher o modelo mais simples, que no caso é o modelo 1

df_coef = pd.DataFrame(modelo_v1.coef_, X.columns, columns = ['Coeficiente'])
print(df_coef.head(10))

#