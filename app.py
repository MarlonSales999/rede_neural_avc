from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Carregar o modelo
filename = 'modelo_final.pkl'
with open(filename, 'rb') as file:  
    modelo_carregado = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    # Obter os dados do pedido
    dados_json = request.json
    dados = pd.DataFrame(dados_json)

    # Remover a coluna 'tipo_residencia' que não é necessária para o modelo
    if 'tipo_residencia' in dados.columns:
        dados = dados.drop(["tipo_residencia"], axis=1)

    # Transformação dos dados categóricos em números
    enc = LabelEncoder()
    for i in dados.columns:
        if dados[i].dtype == "object":
            dados[i] = enc.fit_transform(dados[i])

    # Converter os dados de previsão para um array numpy
    dados_array = dados.values

    # Fazer a previsão
    predicoes = modelo_carregado.predict(dados_array)

    # Retornar as previsões como JSON
    return jsonify(predicoes.tolist())

if __name__ == '__main__':
    app.run(debug=True)
