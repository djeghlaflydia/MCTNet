import folium

# Coordonnées de ta zone (bleu)
blue_south, blue_west, blue_north, blue_east = 32.53, -124.48, 42.01, -114.13

# Approximation de la Californie (rouge)
# (polygone simplifié)
california_coords = [
    [42.01, -124.48],  # Nord-Ouest
    [42.01, -120.00],  # Nord
    [41.50, -119.00],
    [40.00, -120.00],
    [38.50, -123.00],
    [37.00, -122.50],
    [35.00, -121.00],
    [34.00, -119.00],
    [32.53, -117.13],
    [32.53, -114.13],  # Sud-Est
    [34.00, -114.50],
    [36.00, -115.50],
    [38.00, -118.00],
    [40.00, -120.00],
    [42.01, -124.48]   # retour au point de départ
]

# Centre de la carte
center_lat = (blue_south + blue_north) / 2
center_lon = (blue_west + blue_east) / 2

# Création de la carte
m = folium.Map(location=[center_lat, center_lon], zoom_start=6)

# 🔴 Ajouter la Californie (zone rouge)
folium.Polygon(
    locations=california_coords,
    color="red",
    fill=True,
    fill_opacity=0.2,
    tooltip="California"
).add_to(m)

# 🔵 Ajouter ta zone (rectangle bleu)
folium.Rectangle(
    bounds=[[blue_south, blue_west], [blue_north, blue_east]],
    color="blue",
    fill=True,
    fill_opacity=0.3,
    tooltip="Your Selected Area"
).add_to(m)

# Sauvegarder
m.save("california_map.html")

print("Carte générée avec succès ✔️")