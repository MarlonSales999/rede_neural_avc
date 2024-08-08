import requests

url = 'http://127.0.0.1:5000/predict'
dados = [
    {"genero": "Masculino","idade": "93","hipertensao": "1","doenca_cardiaca": "1","casado": "Sim","tipo_trabalho": "Privado","nivel_glicose": "90.2","imc": "20.2","condicao_fumante": "fuma"}
]

response = requests.post(url, json=dados)
print(response.json())