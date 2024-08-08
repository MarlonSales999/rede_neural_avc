import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, Normalizer, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


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

modelo = RandomForestClassifier()

modelo.fit(x_treinamento, y_treinamento)


y_predicao = modelo.predict(x_teste)

print("40% teste e 60% treinamento")
print("\n")
print("Precisões Corretas: %.3f" % accuracy_score(y_teste, y_predicao))
print("Precisões Positivas: %.3f" %
      precision_score(y_teste, y_predicao, average='macro'))
print("Recall: %.3f" % recall_score(y_teste, y_predicao, average='macro'))
print("F1: %.3f" % f1_score(y_teste, y_predicao, average='macro'))

# Código para entender as importâncias de cada parâmetro --- Inicio
nomes_dos_recursos = ['genero','idade','hipertensao','doenca_cardiaca','casado','tipo_trabalho','nivel_glicose','imc','condicao_fumante']
importancias = modelo.feature_importances_
print("Importâncias dos recursos: ", importancias)

# Transformar em DataFrame
importancias_df = pd.DataFrame({'Recurso': nomes_dos_recursos, 'Importancia': importancias})

plt.show()
plt.figure(figsize=(12, 8))
sns.barplot(x='Importancia', y='Recurso', data=importancias_df)
plt.title('Importância dos Recursos')
plt.show()
# Código para entender as importâncias de cada parâmetro --- Fim 

print("\n")
print("30% teste e 70% treinamento")
print("\n")
# Dividindo em conjuntos de treinamento e teste. (30% teste e 70% treinamento)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0)



model = RandomForestClassifier()

# --- Definição de todas as variáveis para utilizar no RandomSearch como parâmetros ---

# Número de árvores na Random Forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 80, num = 10)]

# Número de características a considerar em cada divisão
max_features = ['auto', 'sqrt']

# Número máximo de níveis na árvore
max_depth = [2, 4]

# Número mínimo de amostras necessárias para dividir um nó
min_samples_split = [2, 5]

# Número mínimo de amostras necessárias em cada nó folha
min_samples_leaf = [1, 2]

# Método de seleção de amostras para treinar cada árvore
bootstrap = [True, False]


param_grid = { 
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth' : max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap
}
# --------                    Fim                        ---------

# RandomSearch ---- Inicio.

# GridSearch ---- Inicio.

CV_model = GridSearchCV(estimator=model, param_grid=param_grid, cv= 10, verbose=2, n_jobs=1)
CV_model.fit(x_train, y_train)

model_2=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 10, max_depth=2, bootstrap=True, min_samples_leaf=1, min_samples_split=2)

# GridSearch ---- Fim.

# RandomSearch ---- Inicio.

RG_model = RandomizedSearchCV(estimator = model, param_distributions = param_grid, cv = 10, verbose=2, n_jobs = 1)
RG_model.fit(x_train, y_train)

print(CV_model.best_params_)
print(RG_model.best_params_)

print (f'Train Accuracy - Grid Search : {CV_model.score(x_train,y_train):.3f}')
print (f'Test Accuracy - Grid Search : {CV_model.score(x_test,y_test):.3f}')

print (f'Train Accuracy - Random Search - : {RG_model.score(x_train,y_train):.3f}')
print (f'Test Accuracy - Random Search : {RG_model.score(x_test,y_test):.3f}')

model_3=RandomForestClassifier(random_state=42, max_features='sqrt', n_estimators= 80, max_depth=2, bootstrap=True, min_samples_leaf=2, min_samples_split=2)
# RandomSearch ---- Fim.

# Sem o método para escolher os melhores hiperparâmetros
print('Modelo sem o GridSearch')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

metrics_no_tuning = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, average='macro', zero_division=0),
    "recall": recall_score(y_test, y_pred, average='macro'),
    "f1": f1_score(y_test, y_pred, average='macro')
}

print("Precisões Corretas: %.3f" % accuracy_score(y_test, y_pred))
print("Precisões Positivas: %.3f" %
      precision_score(y_test, y_pred, average='macro', zero_division=0))
print("Recall: %.3f" % recall_score(y_test, y_pred, average='macro'))
print("F1: %.3f" % f1_score(y_test, y_pred, average='macro'))

print("\n")
# ---------------------------------------------------------------------------------------

# Com o método GridSearch
print('Modelo com o GridSearch')
model_2.fit(x_train, y_train)
y_pred = model_2.predict(x_test)

metrics_grid_search = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, average='macro', zero_division=0),
    "recall": recall_score(y_test, y_pred, average='macro'),
    "f1": f1_score(y_test, y_pred, average='macro')
}


print("Precisões Corretas: %.3f" % accuracy_score(y_test, y_pred))
print("Precisões Positivas: %.3f" %
      precision_score(y_test, y_pred, average='macro', zero_division=0 ))
print("Recall: %.3f" % recall_score(y_test, y_pred, average='macro'))
print("F1: %.3f" % f1_score(y_test, y_pred, average='macro'))

print("\n")
# ---------------------------------------------------------------------------------------

# Com o método RandomSearch
print('Modelo com o RandomSearch')
model_3.fit(x_train, y_train)
y_pred = model_3.predict(x_test)

metrics_random_search = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, average='macro', zero_division=0),
    "recall": recall_score(y_test, y_pred, average='macro'),
    "f1": f1_score(y_test, y_pred, average='macro')
}

print("Precisões Corretas: %.3f" % accuracy_score(y_test, y_pred))
print("Precisões Positivas: %.3f" %
      precision_score(y_test, y_pred, average='macro', zero_division=0))
print("Recall: %.3f" % recall_score(y_test, y_pred, average='macro'))
print("F1: %.3f" % f1_score(y_test, y_pred, average='macro'))

print("\n")
# ---------------------------------------------------------------------------------------

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

# Comparar os modelos graficamente
labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
no_tuning_scores = list(metrics_no_tuning.values())
grid_search_scores = list(metrics_grid_search.values())
random_search_scores = list(metrics_random_search.values())

x = np.arange(len(labels))  # Posição das métricas
width = 0.2  # Largura das barras

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, no_tuning_scores, width, label='Sem Tuning')
rects2 = ax.bar(x, grid_search_scores, width, label='GridSearch')
rects3 = ax.bar(x + width, random_search_scores, width, label='RandomSearch')

# Adicionar anotações
ax.set_ylabel('Scores')
ax.set_title('Comparação de Desempenho entre Modelos')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def autolabel(rects):
    """Adiciona os valores das barras"""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('%.3f' % height,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()
plt.show()

import pickle

filename = 'modelo_final.pkl'
with open(filename, 'wb') as file:  
    pickle.dump(model, file)

