import requests

USER_MESSAGE = "hi how are you"

req_msg = { "message":  USER_MESSAGE }

url = "https://amibhavsar.app.n8n.cloud/webhook-test/2a842b2d-2345-4fc1-a399-1838ac2c1da8"

response = requests.post(url, json=req_msg)

print(response.status_code)

print(response.json()[0]["outpput"])