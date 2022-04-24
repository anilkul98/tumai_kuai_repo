import requests
import pandas as pd

data = {'longitude': 29.05240119, 'latitude': 41.19327581}
url = "http://localhost:8080"

response = requests.post(url, json=data, timeout=600)

print(response.json()["response"])