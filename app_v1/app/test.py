import requests

resp = requests.post("http://localhost:5000/predict", files={'file':open('handbook.jpg','rb')})

print(resp.text)