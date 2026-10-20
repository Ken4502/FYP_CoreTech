import urllib.request
import json
from pathlib import Path

boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
img_path = Path('reports/images (1).jfif')
if not img_path.exists():
    img_path = Path('c:/Users/User/Desktop/FYP/CoreTech/Plant_Disease_Experiment/reports/images (1).jfif')

with open(img_path, 'rb') as f:
    img_data = f.read()

body = (
    f'--{boundary}\r\n'
    f'Content-Disposition: form-data; name="file"; filename="{img_path.name}"\r\n'
    f'Content-Type: image/jpeg\r\n\r\n'
).encode('utf-8') + img_data + (
    f'\r\n--{boundary}\r\n'
    f'Content-Disposition: form-data; name="plant"\r\n\r\nUniversal\r\n'
    f'--{boundary}--\r\n'
).encode('utf-8')

req = urllib.request.Request(
    'http://127.0.0.1:8001/api/predict',
    data=body,
    headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
)
try:
    res = urllib.request.urlopen(req)
    data = json.loads(res.read())
    print("STATUS: SUCCESS")
    print("Species Identified:", data["species"]["name"], f"({data['species']['confidence']}%)")
    print("Detected Mode:", data["species"]["detected_mode"])
    print("Diagnosis Condition:", data["diagnosis"]["condition_name"], f"({data['diagnosis']['confidence']}%)")
    print("Is Healthy:", data["diagnosis"]["is_healthy"])
    print("Spots Count:", data["spots"]["spots_count"])
    print("Top Predictions:")
    for p in data["predictions"][:5]:
        print(f"  - #{p['rank']} {p['condition_name']} ({p['class_name']}): {p['confidence']}%")
except Exception as e:
    print("ERROR:", e)
