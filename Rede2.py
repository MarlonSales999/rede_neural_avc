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
from imblearn import FunctionSampler
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler  
from imblearn.pipeline import make_pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
# Carregar dados
dados = pd.read_csv('rede_neural_avc/derrame_cerebral.csv')

dados.info()

# Remover duplicatas
dados.drop_duplicates(inplace=True)


# Dividir dados em X e y
y = dados["avc"]
x = dados.drop("avc", axis=1)

# Import SMOTE
from imblearn.over_sampling import SMOTE

x_encoder = pd.get_dummies(x)

# Perform oversampling with SMOTE
smote = SMOTE()
x_smote, y_smote = smote.fit_resample(x_encoder, y)

# Create a StandardScaler instance
scaler = StandardScaler()

# Fit the StandardScaler
x_scaler = scaler.fit(x_smote)

# Dividir os dados em treino e teste
x_train, x_test, y_train, y_test = train_test_split(x_smote, y_smote, test_size=0.3, random_state=1)

modelo = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=1)

modelo.fit(x_train, y_train)


y_predicao = modelo.predict(x_test)

print("30% teste e 70% treinamento")
print("\n")
print("Precisões Corretas: %.3f" % accuracy_score(y_test, y_predicao))
print("Precisões Positivas: %.3f" %
      precision_score(y_test, y_predicao, average='macro'))
print("Recall: %.3f" % recall_score(y_test, y_predicao, average='macro'))
print("F1: %.3f" % f1_score(y_test, y_predicao, average='macro'))

import pickle

filename = 'modelo_final.pkl'
with open(filename, 'wb') as file:  
    pickle.dump(modelo, file)


filename = 'colunas_treinamento.pkl'
with open(filename, 'wb') as file:  
    pickle.dump(x_encoder, file)