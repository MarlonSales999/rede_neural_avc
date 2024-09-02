import logging
from flask import Flask, request, jsonify, make_response
import pickle
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Carregar o modelo
filename = 'modelo_final.pkl'
with open(filename, 'rb') as file:
    modelo_carregado = pickle.load(file)

# Carregar as colunas usadas no treinamento
with open('./colunas_treinamento.pkl', 'rb') as file:
    colunas_treinamento = pickle.load(file)

# Assegurar que colunas_treinamento é uma lista unidimensional
if isinstance(colunas_treinamento, pd.DataFrame):
    colunas_treinamento = colunas_treinamento.columns.tolist()
elif isinstance(colunas_treinamento, pd.Series):
    colunas_treinamento = colunas_treinamento.tolist()

@app.route('/predict', methods=['POST'])
def predict():
    # Verificar se o tipo de conteúdo é JSON
    if request.content_type != 'application/json':
        return make_response(jsonify({'error': 'Content-Type must be application/json'}), 415)
    
    # Obter os dados do pedido
    try:
        dados_json = request.json
        logging.debug(f"Dados recebidos: {request.json}")
        # Transformar o JSON em um DataFrame
        dados = pd.DataFrame([dados_json])
        dados.info()
    except Exception as e:
        return make_response(jsonify({'error': 'Invalid JSON data', 'message': str(e)}), 400)
    
    # Verificar se há dados no JSON
    if dados.empty:
        return make_response(jsonify({'error': 'Empty JSON data'}), 400)
    
    print(dados.values)
    # Realizar a codificação dos dados novos com as mesmas colunas do treinamento
    x_encoder = pd.get_dummies(dados)
    print(x_encoder.values)
    # Ajustar as colunas dos novos dados para que tenham as mesmas do conjunto de treinamento
    x_encoder = x_encoder.reindex(columns=colunas_treinamento, fill_value=0)
    # Fazer a previsão
    try:
        predicoes = modelo_carregado.predict(x_encoder)
        print(predicoes)
    except Exception as e:
        return make_response(jsonify({'error': 'Prediction error', 'message': str(e)}), 500)

    # # Retornar as previsões como JSON
    return jsonify(predicoes.tolist())
  
if __name__ == '__main__':
    app.run(debug=True)
