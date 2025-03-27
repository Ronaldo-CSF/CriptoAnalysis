# %% [markdown]
# # Previsão de preço de Ethereum e Solana um estudo comparativo de técnicas de Machine Learning

# %% [markdown]
# ## Objetivo
# 
# O presente notebook tem como objetivo demonstrar a implementação de um algoritmo de machine learnig 
# que possa auxiliar a tomada de decisão na compra de criptoativos, por meio da análise de dados históricos
# como preço e captalização de mercado, permitindo identificar padrões e prever movimentações de mercado.
# 
# Para tal será realizada uma análise comparativa da precisão de previsões dos preços das criptomoedas Ether e 
# Sol, utilizando três técnicas distintas de aprendizado de máquina: Random Forest [RF], LSTM e GRU. Pretende-se avaliar a 
# eficiência e acurácia de cada modelo nas previsões, por meio das métricas de Erro Quadrático Médio [MSE], Raiz do Erro 
# Quadrático Médio [RMSE], Erro Absoluto Médio [MAE]; proporcionando uma compreensão sobre o desempenho de cada abordagem.
# 
# 
# ## Estrutura do Trabalho
# 
# 1. Coleta de Dados
# 2. Análise Explorativa de Dados
# 3. Pré-processamento dos Dados
# 4. Aplicação de Técnicas de Machine Learning
# 5. Validação e Teste dos Modelos
# 6. Interpretação dos Resultados

# %% [markdown]
# ### 1. Coleta de Dados
# Recolher dados históricos de preços de diferentes criptoativos, volume de negociação, entre outras variáveis relevantes.

# %% [markdown]
# Os dados utilizados nesse trabalho foram extraidos do notebook presente no Kaggle: [The Most 50 Popular Crypto Data](https://www.kaggle.com/datasets/kaanxtr/btc-price-1m?resource=download).

# %% [markdown]
# 
# ### 2. Análise Explorativa de Dados:
# Realizar uma análise inicial dos dados para entender melhor os padrões históricos e a correlação entre diferentes criptoativos.
# Utilizar visualizações gráficas e estatísticas descritivas para identificar tendências e anomalias.
# 

#%% In[0]: Importação de Pacotes
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from time import time
from grafico import formatar_grafico_mbausp

#%% In[1]: Importar base de dados
eth_data = pd.read_csv("databases/ETHUSDT.csv")
sol_data = pd.read_csv("databases/SOLUSDT.csv")

#%%  In[2.1]: Visualização das 6 primeiras e últimas observações
eth_data.head()

#%% In[2.2]: Visualização das 6 primeiras observações
sol_data.head() 

#%% In[3]: Visualização das variáveis
print(eth_data.keys(),sol_data.keys(), sep="\n\n")

#%% In[4]: Selecionar a Cripto a ser analisada
cripto_lista = {"eth":eth_data, "sol":sol_data}

#%% In[5]: Selecionar as Variáveis a manter
cripto = cripto_lista["eth"][['timestamp', 'close']]

cripto['timestamp'] = pd.to_datetime(cripto['timestamp'])
cripto.set_index('timestamp', inplace=True)

#%% In[6]: Criação de Objeto TimeSeries
cripto_ts = pd.Series(data=cripto['close'].values,index=cripto.index)
print(cripto_ts.head())  # Verifique os primeiros valores

#%% In[7]: Grafico como serie de tempo usando Plotly (Selecione todos os comandos)
formatar_grafico_mbausp(cripto_ts,cripto_ts,titulo_x="Timestamp",
                        titulo_y="Fechamento (ETHUSD)")

#%% In[8]: Preencher os valores faltantes com base no valor mais recente: 
# forward fill

cripto['close'] = cripto['close'].ffill()

# %% In[9]: Realizando o resamplying dos dados

# 30 min
cripto_30min = cripto.resample('30T').mean()
print(cripto_30min)

# 1 hora
cripto_1hr = cripto.resample('1H').mean()
print(cripto_1hr)

# 1 dia
cripto_1dia = cripto.resample('1D').mean()
print(cripto_1dia)

# %% [markdown]

# >Ao analisar as colunas formadas observa-se a presença de valores faltantes, o que prejudica a analise e  
# impede a continuação do código. Como os dados utilizados são de cotações, a última cotação existente é o  
# último preço do ativo. Essa é a própria forma como o preço é apresentado. Assim, iremos preencher esses 
# valores com o último valor presente e as colunas de volume e número de transações com zero.

# In[10]: Preencher Missing Values

# lista de resamples
resample_cripto = [cripto, cripto_30min, cripto_1hr, cripto_1dia]

for df in resample_cripto:
    # Preenchendo a 'close' com o último valor válido
    df["close"] = df["close"].fillna(method='ffill')


# In[11]: Decomposição de Séries Temporais

# Decomposicao pelo modelo ADITIVO
from statsmodels.tsa.seasonal import seasonal_decompose
cripto_decomp = seasonal_decompose(cripto["close"], model='aditive', period=60*24*30*12)

# observando os valores da decomposicao pelo modelo aditivo
print(cripto_decomp.trend)
print(cripto_decomp.seasonal)
print(cripto_decomp.resid)


# In[12]: Plotar a decomposicao (Selecionar todos os comandos)
plt.figure(figsize=(10, 8))
plt.subplot(4, 1, 1)
plt.plot(cripto_decomp.trend)
plt.title('Tendencia')

plt.subplot(4, 1, 2)
plt.plot(cripto_decomp.seasonal)
plt.title('Componente Sazonal')

plt.subplot(4, 1, 3)
plt.plot(cripto_decomp.resid)
plt.title('Residuos')

plt.subplot(4, 1, 4)
plt.plot(cripto_1hr["close"], label='Original')
plt.plot(cripto_decomp.trend + cripto_decomp.seasonal + cripto_decomp.resid, label='Reconstruida')
plt.title('Original vs. Reconstruida')
plt.legend()

plt.tight_layout()
plt.show()

# %% In[13]: Criação de Features Adicionais:

# Random Forest não lida diretamente com a dependência temporal, como outros modelos
# especializados em séries temporais (ARIMA, LSTM). Para contornar isso, é necessário 
# criar variáveis explicativas (X) a partir de transformações nos dados, como janelas 
# deslizantes, indicadores técnicos ou agregados temporais.

for df in resample_cripto:

    # Retornos percentuais (% variação do preço).
    df['return_pct'] = df['close'].pct_change() * 100

    # Médias móveis (curta e longa duração).
    df['SMA_10'] = df['close'].rolling(window=10).mean()  # Média móvel simples (janela de 10 períodos)
    df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()  # Média móvel exponencial (janela de 10 períodos)

    # Índice de Força Relativa (RSI - Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()  # Ganhos médios
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()  # Perdas médias
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df.dropna(inplace=True)


## ---------------------TESTE COM DADOS Diários --------------------- ##

# %% In[14]: Segregar o dataset em dados de treino, validação e teste

# Lembre: resample_cripto = [cripto, cripto_30min, cripto_1hr, cripto_1dia]
df = resample_cripto[3] #cripto_1dia

# Proporções
train_size = int(len(df) * 0.8)  # 80% para treino
# val_size = int(len(df) * 0.2)    # 20% para validação

# Divisão em treino e teste
train = df[:train_size]
# val = df[train_size:train_size+val_size]
test = df[train_size:]

# Verificando as divisões
print(f'Tamanho treino: {len(train)}, teste: {len(test)}')

# Preparação dos dados
X_train, y_train = train.drop(columns=['close']), train['close']
# X_val, y_val = val.drop(columns=['close']), val['close']
X_test, y_test = test.drop(columns=['close']), test['close']


# %% In[15]: Aplicar a normalização dos dados de treino do modelo
from sklearn.preprocessing import StandardScaler

# Escalonar apenas o conjunto de treino
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Usar o mesmo escalador nos outros conjuntos
X_val_scaled = scaler.transform(X_test)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------------------------------------------ #
 # %% In[16_x]: Aplicar o Algoritmo de Return-Based Naive Model usado como benchmark

# from sklearn.metrics import mean_absolute_error, mean_squared_error

# # Fazer previsões no conjunto de teste usando o modelo de persistência com retornos
# test['return_BM'] = train['return_pct'].iloc[-1] / 100  # Use o último retorno conhecido no treino

# # Criando a coluna de valores previstos
# test['predicted_price_BM']=''

# # Copiando o primeiro valor real do conjunto de teste para valores previstos
# test['predicted_price_BM'][0]=test['close'][0]

# #Prevendo com base na fórmula mantendo os valores cumulativos
# test['predicted_price_BM'] = test['predicted_price_BM'].iloc[0] * (1 + test['return_BM']).cumprod()

# # Calcular o erro (exemplo: MAE)
# mae_benchmark_test = mean_absolute_error(test['close'].iloc[1:], test['predicted_price_BM'].iloc[1:])

# mse_benchmark_test = mean_squared_error(y_test, test['predicted_price_BM'])
# rmse_benchmark_test = np.sqrt(mse_benchmark_test)

# print(f'MAE_test do Benchmark: {mae_benchmark_test: .2f}')
# print(f'MSE_test do Benchmark: {mse_benchmark_test: .2f}')
# print(f'RMSE_test do Benchmark: {rmse_benchmark_test: .2f}')

# > Não foi aplicado, resultados inferiores ao Naive tradicional

# %% In[16]: Aplicar o Algoritmoo Naive, considerando para esse algoritmo
# o ultimo valor presente no conjunto de validação

# Tempo inicial
start_time = time()

# Criando a coluna de valores previstos
test['predicted_price_BM2']=test['close'][0]

# Copiando o primeiro valor real do conjunto de teste para valores previstos
test['predicted_price_BM2'][0]=test['close'][0]

# Calcular o erro (exemplo: MAE)
mae_benchmark_test = mean_absolute_error(test['close'].iloc[1:], test['predicted_price_BM2'].iloc[1:])

mse_benchmark_test = mean_squared_error(y_test, test['predicted_price_BM2'])
rmse_benchmark_test = np.sqrt(mse_benchmark_test)

print(f'MAE_test do Benchmark: {mae_benchmark_test: .2f}')
print(f'MSE_test do Benchmark: {mse_benchmark_test: .2f}')
print(f'RMSE_test do Benchmark: {rmse_benchmark_test: .2f}')

# Calculo de Tempo de Execução
end_time = time()
execution_time = end_time - start_time

if df is resample_cripto[3]:  
    Naive_result_dia = [mae_benchmark_test, mse_benchmark_test, rmse_benchmark_test,'NA',
                execution_time]
elif df is resample_cripto[2]:
    Naive_result_hr = [mae_benchmark_test, mse_benchmark_test, rmse_benchmark_test,'NA',
                execution_time]
elif df is resample_cripto[1]:
    Naive_result_30min = [mae_benchmark_test, mse_benchmark_test, rmse_benchmark_test,'NA',
                execution_time]
elif df is resample_cripto[0]:
    Naive_result_1min = [mae_benchmark_test, mse_benchmark_test, rmse_benchmark_test,'NA',
                execution_time]

print(f'O tempo que o modelo levou para treinar foi de: {execution_time}')

# %% [markdown]

# ### Modelagem
# A partir desse momento serão apresentados os algoritmos a serem utilizados nesse
# estudo.
# #### 1. Random Forest

# %% In[17]: Aplicar o Algoritmo de Random Forest

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from time import time

# %% In[17]: Random Forest Puro

start_time = time()

# Modelo Random Forest
rf_model = RandomForestRegressor(n_estimators=300, 
                                 random_state=42,
                                 criterion='squared_error',
                                 n_jobs=-1) #max_depth=10 - testar

# Treinamento
rf_model.fit(X_train_scaled, y_train)

# Previsão no conjunto de teste
y_test_pred = rf_model.predict(X_test_scaled)

# Avaliação no conjunto de teste
mae_rf_test = mean_absolute_error(y_test, y_test_pred)
mse_rf_test = mean_squared_error(y_test, y_test_pred)
rmse_rf_test = np.sqrt(mse_rf_test)

print(f'MAE test: {mae_rf_test:.2f}')
print(f'MSE test: {mse_rf_test:.2f}')
print(f'RMSE test: {rmse_rf_test:.2f}')

# Calculo do incicador MASE
mase_rf = mae_rf_test/mae_benchmark_test
print(f'MASE para RF: {mase_rf: .2f}')

end_time = time()
execution_time = end_time - start_time
print(f'O tempo que o modelo levou para treinar foi de: {execution_time}')

if df is resample_cripto[3]:  
    RF_result_dia = [mae_rf_test, mse_rf_test, rmse_rf_test,mase_rf,
                execution_time]
elif df is resample_cripto[2]:
    RF_result_hr = [mae_rf_test, mse_rf_test, rmse_rf_test,mase_rf,
                execution_time]
elif df is resample_cripto[1]:
    RF_result_30min = [mae_rf_test, mse_rf_test, rmse_rf_test,mase_rf,
                execution_time]
elif df is resample_cripto[0]:
    RF_result_1min = [mae_rf_test, mse_rf_test, rmse_rf_test,mase_rf,
                execution_time]

# ------------------------------------------------------------------------------

#%% In[18]: Random Forest com GridSearch

start_time = time()

param_grid = {
    "n_estimators": [100, 300, 500, 1000],
    "max_features": ["sqrt",None], # Como sqrt(4)=log2(4), não faz sentido variar
    "max_depth": [10, 20, 100, None],
    "min_samples_leaf":[1,5],
    }

rf_model = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(estimator=rf_model, 
                           param_grid=param_grid, 
                           scoring='neg_root_mean_squared_error', 
                           cv=4, 
                           n_jobs=-1)

grid_search.fit(X_train, y_train.values)

# Print the best parameters and the best score
print(grid_search)
print(grid_search.best_params_)
print(grid_search.best_score_)
end_time = time()
execution_time = end_time - start_time
print(f'O tempo que o modelo levou para treinar foi de: {execution_time}')

grid_results = grid_search.cv_results_

 #%% Rodando o melhor modelo
start_time = time()

# SOL - 1dia parametros: {n_estimators= 1000, max_features= 'sqrt', max_depth= 20,
#                           min_samples_leaf= 1,random_state=42,n_jobs=-1}

# SOL - 1hr parametros: {n_estimators= 100, max_features= 'sqrt', max_depth= 100,
#                           min_samples_leaf= 1,random_state=42,n_jobs=-1}

# SOL - 30min parametros: {n_estimators= 1000, max_features= 'sqrt', max_d epth= 20,
#                           min_samples_leaf= 1,random_state=42,n_jobs=-1}

# ETHER - 1dia parametros: {n_estimators= 300, max_features= 'sqrt', max_depth= 20,
#                           min_samples_leaf= 1,random_state=42,n_jobs=-1}

# ETHER - 1hr parametros: {n_estimators= 1000, max_features= 'sqrt', max_depth= 20,
#                           min_samples_leaf= 1,random_state=42,n_jobs=-1}

# ETHER - 30min parametros: {n_estimators= 1000, max_features= None, max_depth= None,
#                           min_samples_leaf= 1,random_state=42,n_jobs=-1}


best_rf_model = RandomForestRegressor(n_estimators = 1000,
                                      max_features = None,
                                      max_depth = None,
                                      min_samples_leaf = 1,
                                      random_state = 42,
                                      n_jobs = -1)

best_rf_model.fit(X_train, y_train.values)

# Previsão no conjunto de teste
y_test_pred = best_rf_model.predict(X_test_scaled)

# Avaliação no conjunto de teste
mae_rf_test = mean_absolute_error(y_test, y_test_pred)
mse_rf_test = mean_squared_error(y_test, y_test_pred)
rmse_rf_test = np.sqrt(mse_rf_test)

print(f'MAE test: {mae_rf_test:.2f}')
print(f'MSE test: {mse_rf_test:.2f}')
print(f'RMSE test: {rmse_rf_test:.2f}')

# Calculo do incicador MASE
mase_rf = mae_rf_test/mae_benchmark_test
print(f'MASE para RF: {mase_rf: .2f}')

end_time = time()
execution_time = end_time - start_time
print(f'O tempo que o modelo levou para treinar foi de: {execution_time}')

if df is resample_cripto[3]:  
    best_RF_result_dia = [mae_rf_test, mse_rf_test, rmse_rf_test,mase_rf,
                execution_time]
elif df is resample_cripto[2]:
    best_RF_result_hr = [mae_rf_test, mse_rf_test, rmse_rf_test,mase_rf,
                execution_time]
elif df is resample_cripto[1]:
    best_RF_result_30min = [mae_rf_test, mse_rf_test, rmse_rf_test,mase_rf,
                execution_time]
elif df is resample_cripto[0]:
    best_RF_result_1min = [mae_rf_test, mse_rf_test, rmse_rf_test,mase_rf,
                execution_time]













# %% [markdown]
# >Como temos que RMSE_RF < RMSE_benchmark e MASE_RF < 1 concluimos que o modelo de RF é superior ao 
# Modelo de Persistência com Retornos usado como benchmarket nesse trabalho


# %% In[19]: Visualizando os resultados do modelo:

# Previsões completas: treino e teste
y_train_pred = rf_model.predict(X_train_scaled)
y_test_pred = rf_model.predict(X_test_scaled)

# Concatenando os valores reais e previstos
y_real = pd.concat([y_train, y_test])
y_pred = pd.concat([
    pd.Series(y_train_pred, index=y_train.index),
    pd.Series(y_test_pred, index=y_test.index)])

plt.figure(figsize=(10, 6))

# Plotando os valores reais com as datas no eixo x
plt.plot(y_real.index, y_real, label='Série Original (Valores Reais)', color = 'blue', alpha=0.7)

# Plotando as previsões com as mesmas datas
plt.plot(y_pred.index, y_pred, label='Previsões (treino+teste)', color = 'orange', alpha=0.7)

# Ajustando o gráfico
plt.legend()
plt.title('Série Temporal: Original vs Previsões (Random Forest)')
plt.xlabel('Data')  # Rótulo do eixo x
plt.ylabel('Valor de Fechamento')  # Rótulo do eixo y
plt.grid(True)

# Rotação das datas para melhor visualização
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3. Pré-processamento dos Dados: 
# Limpeza dos dados, tratamento de valores ausentes e normalização dos dados para garantir a qualidade e consistência das análises subsequentes.

# %% [markdown]
# ### 4. Aplicação de Técnicas de Machine Learning:
# Modelos de Regressão: Utilizar modelos como Regressão Linear para prever preços futuros com base em tendências históricas.
# Modelos de Classificação: Aplicar algoritmos como Random Forest e SVM para classificar os períodos de alta e baixa no mercado.
# 

# %% [markdown]
# ### 5. Validação e Teste dos Modelos:
# Dividir os dados em conjuntos de treinamento e teste para validar a eficácia dos modelos.
# Utilizar técnicas de validação cruzada para garantir a robustez dos resultados.
# 

# %% [markdown]
# ### 6. Interpretação dos Resultados:
# Analisar o desempenho dos modelos, identificando quais técnicas apresentam melhores resultados na previsão de ciclos de mercado.
# Discussão sobre as limitações do estudo e possíveis melhorias para pesquisas futuras.







# Inicialmente proposto:
#    1. Base de dados:
#       sol = sol_data[['timestamp', 'close', 'volume', 'number_of_trades']]
#    2. Resample funcions:
#       cripto_5min = cripto.resample('5T').agg({'close': 'mean', 'volume': 'sum', 'number_of_trades': 'sum'})
#    3. Criação do Objeto Serie do Resample:
#       cripto_close_5m = pd.Series(data=cripto_5min['close'].values, index=cripto_5min.index)
#    4. Preencher Missing Value:
#       for df in resample_cripto:
#           df[["volume", "number_of_trades"]] = df[["volume", "number_of_trades"]].fillna(0)
#    5. Preparação dos dados para Random Forest 
#       X = cripto_1hr_stand[['volume', 'number_of_trades', 'return_pct', 'SMA_10', 'EMA_10', 'RSI']] 