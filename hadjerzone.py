import folium

# Créer la carte centrée entre les deux zones
m = folium.Map(location=[37, -100], zoom_start=4)

# ------------------------
# Arkansas (bbox)
# ------------------------
arkansas_bounds = [[33.00, -94.62], [36.50, -89.64]]

folium.Rectangle(
    bounds=arkansas_bounds,
    color='red',
    fill=True,
    fill_opacity=0.3,
    popup="Arkansas"
).add_to(m)

# ------------------------
# Californie (bbox)
# ------------------------
california_bounds = [[32.53, -124.48], [42.01, -114.13]]

folium.Rectangle(
    bounds=california_bounds,
    color='blue',
    fill=True,
    fill_opacity=0.3,
    popup="Californie"
).add_to(m)

# Ajuster automatiquement le zoom pour voir toutes les zones
m.fit_bounds([
    [32.53, -124.48],
    [42.01, -89.64]
])

# Sauvegarder en HTML
m.save("map.html")

print("✅ Fichier 'map.html' créé ! Ouvre-le dans ton navigateur.")