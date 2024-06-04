import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


dados = pd.read_csv('derrame_cerebral.csv')
print("\n")
# Análise Exploratória do Banco de Dados
dados.info()
# --> tipo_residencia não faz muito sentido para o modelo.

dados['tipo_residencia']

# Então Dropamos.
dados = dados.drop(["tipo_residencia"], axis=1)

# Removemos dados duplicados se houverem.
dados.drop_duplicates

# Transformação dos dados categóricos em números para que possam ser usados em algoritmos de aprendizado de máquina, colunas de texto viraram números, Ex: Feminimo e Masculino -> 0 e 1.

enc = LabelEncoder()
for i in dados.columns:
    if dados[i].dtype == "object":
        dados[i] = enc.fit_transform(dados[i])

dados.info()

# Mapa de calor para analisar correlações e tentar tirar alguma informação.

sns.heatmap(dados.corr())

# Mostra o gráfico de calor

plt.show()

#  Isso isola a coluna avc do DataFrame, ou seja, a variável alvo.

y = dados["avc"]
x = dados.drop(["avc"], axis=1)

# Preparando os dados para análise ou modelagem, garantindo que os recursos estejam na mesma escala e que todas as amostras tenham o mesmo comprimento de vetor.
scaler = StandardScaler()
x = scaler.fit_transform(x)
x = Normalizer().fit_transform(x)

# Dividindo em conjuntos de treinamento e teste. (40% teste e 60% treinamento)
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(
    x, y, test_size=0.40, random_state=0)

modelo = RandomForestClassifier(random_state=0)
modelo.fit(x_treinamento, y_treinamento)
y_predicao = modelo.predict(x_teste)

print("40% teste e 70% treinamento")
print("\n")
print("Precisões Corretas: %.3f" % accuracy_score(y_teste, y_predicao))
print("Precisões Positivas: %.3f" %
      precision_score(y_teste, y_predicao, average='macro'))
print("Recall: %.3f" % recall_score(y_teste, y_predicao, average='macro'))
print("F1: %.3f" % f1_score(y_teste, y_predicao, average='macro'))


print("\n")
print("30% teste e 70% treinamento")
print("\n")
# Dividindo em conjuntos de treinamento e teste. (30% teste e 70% treinamento)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0)

model = RandomForestClassifier(random_state=0)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("Precisões Corretas: %.3f" % accuracy_score(y_test, y_pred))
print("Precisões Positivas: %.3f" %
      precision_score(y_test, y_pred, average='macro'))
print("Recall: %.3f" % recall_score(y_test, y_pred, average='macro'))
print("F1: %.3f" % f1_score(y_test, y_pred, average='macro'))

print("\n")

# Lista das acurácias
acuracias = [0.949, 0.953]

# Lista de treinamento e teste
quebras = ['60-40', '70-30']

# Irá fazer um gráfico com a precisão do desempenho de cada modelo em relação à divisão do conjunto de treinamento e teste
plt.plot(quebras, acuracias)
plt.xlabel('Divisão do conjunto de treinamento e teste')
plt.ylabel('Precisão de desempenho')
plt.title('Precisão de desempenho de modelos de Random Forest com diferentes divisões de conjuntos de treinamento e teste')
plt.show()


pontuacao_precisao = [0.477, 0.478]
pontuacao_recall = [0.498, 0.499]
pontuacao_f1 = [0.487, 0.488]
cores = ['blue', 'red']
labels = ['60-40 split', '70-30 split']

fig, ax = plt.subplots()

# Defina os rótulos do eixo x para serem as divisões de teste de treinamentoax.set_xticks([0, 1])
ax.set_xticks([0, 1])  # Defina as posições dos ticks
ax.set_xticklabels(labels)

# Plota as pontuações de precisão, Recall score e F1 score.
ax.plot(pontuacao_precisao, label='Precisão', color=cores[0])
ax.plot(pontuacao_recall, label='Recall', color=cores[1])
ax.plot(pontuacao_f1, label='F1', color='green')

# Título do gráfico
ax.set_title('Precisão, recall e F1 de modelos de floresta aleatórios com diferentes divisões de conjuntos de treinamento e teste')
ax.set_xlabel('Divisão de treinamento-teste')
ax.set_ylabel('Pontuação')

# Legenda para o gráfico
ax.legend()

# Mostra o gráfico
plt.show()
