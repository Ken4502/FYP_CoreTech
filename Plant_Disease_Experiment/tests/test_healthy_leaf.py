import urllib.request
import json
import glob
from pathlib import Path

# Test healthy samples from each supported class
for grp in ['Money_Plant_Healthy', 'Aloe_Healthy', 'Cactus_Healthy']:
    imgs = glob.glob(f'data/test/{grp}/*.*')
    if not imgs:
        continue
    img_path = Path(imgs[0])
    boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
    with open(img_path, 'rb') as f:
        img_data = f.read()

    body = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="file"; filename="{img_path.name}"\r\n'
        f'Content-Type: image/jpeg\r\n\r\n'
    ).encode('utf-8') + img_data + (
        f'\r\n--{boundary}\r\n'
        f'Content-Disposition: form-data; name="plant"\r\n\r\nauto\r\n'
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
        print(f"[{grp}] -> Species: {data['species']['name']} ({data['species']['confidence']}%)")
        print(f"       -> Verdict: {data['diagnosis']['condition_name']} (Healthy={data['diagnosis']['is_healthy']})")
        print(f"       -> Spots: {data['spots']['spots_count']}")
    except Exception as e:
        print(f"[{grp}] ERROR: {e}")

