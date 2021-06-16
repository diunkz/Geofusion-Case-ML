#importando todas as bibliotecas necessárias no projeto

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import statsmodels.api as sm
import os

#importando dados (csv salvo no github)
df_dados = pd.read_csv("https://raw.githubusercontent.com/hypnoh/Geofusion-Case-ML/main/DesafioEstagioMachineLearning.csv")

#colunas sem variáveis categóricas e faturamento
colunasSemFaturamento = ['população', 'popAte9', 'popDe10a14', 'popDe15a19', 'popDe20a24', 'popDe25a34', 'popDe35a49', 'popDe50a59',
                         'popMaisDe60', 'domiciliosA1', 'domiciliosA2', 'domiciliosB1', 'domiciliosB2', 'domiciliosC1', 'domiciliosC2',
                         'domiciliosD', 'domiciliosE', 'rendaMedia']

#mostrando boxplots de cada variável numérica (removendo o faturamento)
for x in colunasSemFaturamento:
  plt.figure(figsize = (16,10))
  df_dados.boxplot(column=x)
  plt.show()

#observando gráficos com dados x faturamento
sns.pairplot(df_dados.drop(columns=["codigo"]), x_vars = ["faturamento"], height=5)

copiaDfDados = df_dados.copy()

#replace 0 com NaN
copiaDfDados[colunasSemFaturamento] = copiaDfDados[colunasSemFaturamento].replace(0, np.nan)

#drop NaN
copiaDfDados.dropna(inplace=True)

#print de todos os valores
'''
for i in X_standard:
  for z in X_standard[i]:
    print(z)
'''

#Linear Regression

rQuadrado = np.array([]);

for x in colunasSemFaturamento:
  variavelASerUtilizada = x

  X = copiaDfDados["faturamento"].values.reshape(-1,1)
  Y = copiaDfDados[variavelASerUtilizada].values.reshape(-1,1)
  lr = LinearRegression()
  lr.fit(X,Y)
  Y_pred = lr.predict(X)

  X1 = copiaDfDados[variavelASerUtilizada]
  y1 = copiaDfDados['faturamento']
  X2 = sm.add_constant(X1)
  est = sm.OLS(y1, X2)
  est2 = est.fit()
  rQuadrado = np.append(rQuadrado, est2.rsquared)

#criando tabela para verificacao do R2

colunas = pd.DataFrame()
colunas.insert(loc = 0, column="variável", value=colunasSemFaturamento)
colunas.insert(loc = 1, column="valor R2", value=rQuadrado)

#mostrando os valores de R2 mais próximos de 1
colunas = colunas.sort_values(by=["valor R2"], ascending = False)
colunas = colunas.reset_index()
colunas = colunas.drop(columns="index")

#salvando as variáveis dos 3 maiores valores R2
tresMaioresR2 = []

for x in range(3):
  tresMaioresR2.append(colunas["variável"].iloc[x])

-
#Linear Regression das 3 primeiras
for x in tresMaioresR2:
    variavelASerUtilizada = x
    
    X = copiaDfDados["faturamento"].values.reshape(-1,1)
    Y = copiaDfDados[variavelASerUtilizada].values.reshape(-1,1)
    lr = LinearRegression()
    lr.fit(X,Y)
    Y_pred = lr.predict(X)

    #visualizando resultados
    print(f"-> {x} ")
  
    plt.figure(figsize = (16,8))
    plt.scatter(X,Y)
    plt.plot(X, Y_pred, color="red")
    plt.show()

    print("O modelo é: Vendas = {:.5} + {:.5}X\n".format(lr.intercept_[0], lr.coef_[0][0]))

    X1 = copiaDfDados[variavelASerUtilizada]
    y1 = copiaDfDados['faturamento']
    X2 = sm.add_constant(X1)
    est = sm.OLS(y1, X2)
    est2 = est.fit()
    rQuadrado = np.append(rQuadrado, est2.rsquared)
    print(est2.summary())

    print("\n**************************************************************************************************************************\n")

#utilização do StandardScaler()

'''
X_standard = df_dados.drop(columns=['faturamento']).copy()
X_standard [colunasSemFaturamento] = StandardScaler().fit_transform(df_dados[colunasSemFaturamento])
'''

#observando gráficos com dados x faturamento após utilizar StandardScaler()'''
'''
sns.pairplot(X_standard, x_vars = ["faturamento"], height=5)
'''

###

# plotando o grafico variavel x faturamento
'''
plt.figure(figsize = (20,10))
plt.scatter(
    X_standard['faturamento'], 
    X_standard['variavel'], 
    c='red')
plt.xlabel(" ($) Faturamento")
plt.ylabel(" ($) Variável")

plt.show()
'''

