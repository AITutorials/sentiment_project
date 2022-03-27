import requests
url = "http://0.0.0.0:5003/v1/model_prediction/" 
sentence1 = "Oh, my god"
sentence2 = "Fuck!"

data = {"sentence1": sentence1, "sentence2": sentence2}
res = requests.post(url, json=data, timeout=200)
print(res.text)
