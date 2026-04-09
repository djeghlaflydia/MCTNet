import folium
import json

# Coordonnées de la bounding box de la Californie
bbox = [-124.48, 32.53, -114.13, 42.01]  # [west, south, east, north]

# Centre approximatif de la Californie
center_lat = (bbox[1] + bbox[3]) / 2  # (32.53 + 42.01) / 2 = 37.27
center_lon = (bbox[0] + bbox[2]) / 2  # (-124.48 + -114.13) / 2 = -119.305

# Créer la carte
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=6,
    tiles='OpenStreetMap'
)

# 1. Ajouter la bounding box en bleu
rectangle = folium.Rectangle(
    bounds=[[bbox[1], bbox[0]], [bbox[3], bbox[2]]],  # [[south, west], [north, east]]
    color='blue',
    weight=3,
    fill=True,
    fill_opacity=0.1,
    popup='California Bounding Box'
)
rectangle.add_to(m)

# 2. Ajouter un polygone pour toute la Californie (approximation avec un rectangle arrondi)
# Pour une représentation plus précise, on peut utiliser un polygone qui suit approximativement les contours
california_polygon = folium.Polygon(
    locations=[
        [42.01, -124.48],  # Nord-Ouest
        [42.01, -120.00],  # Nord
        [41.00, -120.00],
        [40.00, -122.00],
        [39.00, -122.00],
        [38.00, -120.00],
        [37.00, -119.00],
        [36.00, -119.00],
        [35.00, -116.00],
        [34.00, -117.00],
        [33.00, -117.00],
        [32.53, -117.00],  # Sud
        [32.53, -114.13],  # Sud-Est
        [35.00, -114.50],
        [36.00, -115.00],
        [37.00, -116.00],
        [38.00, -118.00],
        [39.00, -120.00],
        [40.00, -122.00],
        [42.01, -122.00],
        [42.01, -124.48],  # Retour au Nord-Ouest
    ],
    color='red',
    weight=2,
    fill=True,
    fill_color='red',
    fill_opacity=0.3,
    popup='Californie'
)
california_polygon.add_to(m)

# 3. Alternative : utiliser un fichier GeoJSON pour plus de précision
# Note: Pour une carte plus précise, décommentez le code ci-dessous
"""
geojson_data = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-124.48, 42.01],
                    [-124.20, 41.80],
                    [-124.00, 41.50],
                    [-123.80, 41.00],
                    [-123.50, 40.50],
                    [-123.00, 40.00],
                    [-122.50, 39.50],
                    [-122.00, 39.00],
                    [-121.50, 38.50],
                    [-121.00, 38.00],
                    [-120.50, 37.50],
                    [-120.00, 37.00],
                    [-119.50, 36.50],
                    [-119.00, 36.00],
                    [-118.50, 35.50],
                    [-118.00, 35.00],
                    [-117.50, 34.50],
                    [-117.00, 34.00],
                    [-116.50, 33.50],
                    [-116.00, 33.00],
                    [-115.50, 32.53],
                    [-114.13, 32.53],
                    [-114.50, 33.00],
                    [-115.00, 33.50],
                    [-115.50, 34.00],
                    [-116.00, 34.50],
                    [-116.50, 35.00],
                    [-117.00, 35.50],
                    [-117.50, 36.00],
                    [-118.00, 36.50],
                    [-118.50, 37.00],
                    [-119.00, 37.50],
                    [-119.50, 38.00],
                    [-120.00, 38.50],
                    [-120.50, 39.00],
                    [-121.00, 39.50],
                    [-121.50, 40.00],
                    [-122.00, 40.50],
                    [-122.50, 41.00],
                    [-123.00, 41.50],
                    [-123.50, 41.80],
                    [-124.00, 42.00],
                    [-124.48, 42.01]
                ]]
            }
        }
    ]
}

folium.GeoJson(
    geojson_data,
    style_function=lambda x: {
        'color': 'red',
        'weight': 2,
        'fillColor': 'red',
        'fillOpacity': 0.3
    },
    popup='Californie (précis)'
).add_to(m)
"""

# Ajouter un titre
title_html = '''
<div style="position: fixed; 
            top: 10px; left: 50px; 
            width: 300px; height: 40px; 
            background-color: white;
            border: 2px solid black;
            border-radius: 5px;
            z-index: 1000;
            text-align: center;
            font-family: Arial, sans-serif;
            font-weight: bold;
            padding: 5px;">
    <span style="color: blue;">■ Bounding Box</span> | 
    <span style="color: red;">■ Californie</span>
</div>
'''
m.get_root().html.add_child(folium.Element(title_html))

# Ajouter les coordonnées de la bounding box
info_html = '''
<div style="position: fixed; 
            bottom: 10px; left: 50px; 
            width: auto; height: auto; 
            background-color: white;
            border: 1px solid black;
            border-radius: 5px;
            z-index: 1000;
            font-family: Arial, sans-serif;
            font-size: 12px;
            padding: 5px;">
    <b>Bounding Box Californie:</b><br>
    West: -124.48° | South: 32.53°<br>
    East: -114.13° | North: 42.01°
</div>
'''
m.get_root().html.add_child(folium.Element(info_html))

# Sauvegarder la carte
m.save('californie_map.html')
print("✅ Fichier 'californie_map.html' généré avec succès !")
print("📍 Bounding Box: [-124.48, 32.53, -114.13, 42.01]")
print("🔵 Bounding Box affichée en bleu")
print("🔴 Californie affichée en rouge")
print("\nOuvrez le fichier 'californie_map.html' dans votre navigateur pour visualiser la carte.")