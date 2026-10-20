import urllib.request
import urllib.parse
import json

taxa = [
    ('Thrips (Thripidae)', 'Thripidae'),
    ('Whiteflies (Aleyrodidae)', 'Aleyrodidae'),
    ('Fungus Gnats (Sciaridae)', 'Sciaridae'),
    ('Armored Scales (Diaspididae)', 'Diaspididae'),
    ('Botrytis Gray Mold', 'Botrytis cinerea'),
    ('Anthracnose (Colletotrichum)', 'Colletotrichum'),
    ('Alternaria Leaf Spot', 'Alternaria'),
    ('Rust Fungi (Pucciniales)', 'Pucciniales'),
    ('Rose Black Spot (Diplocarpon rosae)', 'Diplocarpon rosae'),
    ('Downy Mildew (Peronosporaceae)', 'Peronosporaceae'),
    ('Late Blight / Rot (Phytophthora)', 'Phytophthora'),
    ('Leaf Miners (Agromyzidae)', 'Agromyzidae'),
    ('Citrus Leafminer (Phyllocnistis citrella)', 'Phyllocnistis citrella'),
    ('Ficus Leaf-Gall Thrips (Gynaikothrips ficorum)', 'Gynaikothrips ficorum'),
    ('Palm Aphid (Cerataphis brasiliensis)', 'Cerataphis brasiliensis'),
    ('Oleander/Wax Plant Aphid (Aphis nerii)', 'Aphis nerii'),
    ('Citrus Snow Scale (Unaspis citri)', 'Unaspis citri'),
    ('Juniper/Apple Rust (Gymnosporangium)', 'Gymnosporangium'),
    ('Powdery Mildew on Begonia/Cucurbits (Podosphaera)', 'Podosphaera')
]

print("--- iNaturalist Plant Pathology Taxa Query ---")
for name, q in taxa:
    try:
        url = f"https://api.inaturalist.org/v1/taxa?q={urllib.parse.quote(q)}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        res = urllib.request.urlopen(req)
        d = json.loads(res.read())
        if d['results']:
            r = d['results'][0]
            obs_cnt = r.get('observations_count', 0)
            print(f"{name:<35} | Taxon ID: {r['id']:<7} | Observations: {obs_cnt:>7}")
    except Exception as e:
        print(f"{name:<35} | Error: {e}")
