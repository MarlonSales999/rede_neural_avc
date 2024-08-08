import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Carregar o modelo
filename = 'modelo_final.pkl'
with open(filename, 'rb') as file:  
    modelo_carregado = pickle.load(file)

# Carregar os novos dados
dados = pd.read_csv('novos_dados.csv')

# Remover a coluna 'tipo_residencia' que não é necessária para o modelo
dados = dados.drop(["tipo_residencia"], axis=1)

# Transformação dos dados categóricos em números para que possam ser usados em algoritmos de aprendizado de máquina
enc = LabelEncoder()
for i in dados.columns:
    if dados[i].dtype == "object":
        dados[i] = enc.fit_transform(dados[i])

dados.info()

# Converter os dados de previsão para um array numpy para evitar problemas de nomes de recursos
dados_array = dados.values

# Fazer a previsão
predicoes = modelo_carregado.predict(dados_array)

print(predicoes)
