"""
=============================================================================
  CARTOGRAPHIE DES CULTURES - Collecte de données Sentinel-2 via GEE
  Basé sur : MCTNet (Wang et al., 2024) - Computers and Electronics in Agriculture
=============================================================================

DESCRIPTION :
  Ce script collecte des données Sentinel-2 (10 bandes, 36 périodes de 10 jours)
  pour l'année 2021 sur deux zones : Arkansas et Californie.
  Les données sont exportées au format CSV vers Google Drive.

PRÉREQUIS :
  pip install earthengine-api geemap pandas numpy matplotlib

AUTHENTIFICATION :
  1. Créer un projet Google Cloud : https://console.cloud.google.com
  2. Activer l'API Earth Engine dans le projet
  3. Lancer : earthengine authenticate
  4. Ou dans Colab : ee.Authenticate() puis ee.Initialize(project='votre-projet')

STRUCTURE DES DONNÉES PRODUITES :
  Chaque ligne du CSV = 1 pixel labellisé
  Colonnes : [latitude, longitude, crop_type] + [B2_t1, B3_t1, ..., B12_t36]
  Soit 360 features spectrales (10 bandes × 36 périodes)
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1. IMPORTS ET AUTHENTIFICATION
# ─────────────────────────────────────────────────────────────────────────────
import ee
import geemap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta

# ── Authentification GEE ──────────────────────────────────────────────────────
# Option A : environnement local (terminal)
#   earthengine authenticate
#   ee.Initialize(project='votre-projet-gcloud')

# Option B : Google Colab (décommenter les 2 lignes suivantes)
# ee.Authenticate()
# ee.Initialize(project='votre-projet-gcloud')

# Initialisation standard (remplacer par votre projet GCloud)
try:
    ee.Initialize(project='your-gcloud-project-id')
    print("✅ Google Earth Engine initialisé avec succès.")
except Exception as e:
    print(f"❌ Erreur d'initialisation GEE : {e}")
    print("   → Lancez d'abord : earthengine authenticate")
    raise


# ─────────────────────────────────────────────────────────────────────────────
# 2. DÉFINITION DES ZONES D'ÉTUDE
#    Coordonnées issues de la Figure 1 de l'article MCTNet
# ─────────────────────────────────────────────────────────────────────────────

# ── Arkansas : zone agricole intensive (riz, soja, maïs, coton, blé) ─────────
ARKANSAS_BBOX = ee.Geometry.Rectangle(
    [-94.62, 33.00, -89.64, 36.50],   # [lon_min, lat_min, lon_max, lat_max]
    proj='EPSG:4326',
    evenOdd=False
)

# Mapping area 1 (nord) et area 2 (sud) pour l'évaluation cartographique
ARKANSAS_MAP_AREA_1 = ee.Geometry.Rectangle([-92.5, 35.0, -91.5, 36.0])
ARKANSAS_MAP_AREA_2 = ee.Geometry.Rectangle([-91.5, 33.5, -90.5, 34.5])

# ── Californie : zone agricole diversifiée (raisins, riz, luzerne, amandes...) ─
CALIFORNIA_BBOX = ee.Geometry.Rectangle(
    [-124.48, 32.53, -114.13, 42.01],
    proj='EPSG:4326',
    evenOdd=False
)

# Mapping area 1 (nord, Sacramento Valley) et area 2 (San Joaquin Valley)
CALIFORNIA_MAP_AREA_1 = ee.Geometry.Rectangle([-122.5, 38.5, -121.5, 39.5])
CALIFORNIA_MAP_AREA_2 = ee.Geometry.Rectangle([-120.5, 36.5, -119.5, 37.5])

# Dictionnaire des zones pour itération
STUDY_AREAS = {
    'Arkansas': {
        'bbox':     ARKANSAS_BBOX,
        'map_area_1': ARKANSAS_MAP_AREA_1,
        'map_area_2': ARKANSAS_MAP_AREA_2,
        # Classes CDL présentes dans l'article (Table 2)
        'crop_classes': {
            'Soybeans': 5,    # CDL code
            'Corn':     1,
            'Cotton':   2,
            'Rice':     3,
            'Others':   0     # agrégation des classes < 5 %
        }
    },
    'California': {
        'bbox':     CALIFORNIA_BBOX,
        'map_area_1': CALIFORNIA_MAP_AREA_1,
        'map_area_2': CALIFORNIA_MAP_AREA_2,
        'crop_classes': {
            'Grapes':    69,
            'Rice':       3,
            'Alfalfa':   36,
            'Almonds':   75,
            'Pistachios': 204,
            'Others':     0
        }
    }
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. PARAMÈTRES SENTINEL-2 ET TEMPORELS
#    Conformément à la Section 2.2.3 et 2.2.4 de l'article
# ─────────────────────────────────────────────────────────────────────────────

# 10 bandes sélectionnées (Band 1, 9, 10 exclues : résolution 60 m)
S2_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
BAND_NAMES = {
    'B2':  'Blue',
    'B3':  'Green',
    'B4':  'Red',
    'B5':  'RedEdge1',
    'B6':  'RedEdge2',
    'B7':  'RedEdge3',
    'B8':  'NIR',
    'B8A': 'RedEdge4',
    'B11': 'SWIR1',
    'B12': 'SWIR2'
}

# Génération des 36 périodes de 10 jours (2021)
def generate_10day_periods(year=2021):
    """
    Génère 36 intervalles de ~10 jours couvrant l'année 2021.
    Retourne une liste de tuples (date_début, date_fin) en format 'YYYY-MM-DD'.
    """
    periods = []
    start = datetime(year, 1, 1)
    end   = datetime(year, 12, 31)
    current = start
    while current <= end:
        period_end = min(current + timedelta(days=9), end)
        periods.append((
            current.strftime('%Y-%m-%d'),
            period_end.strftime('%Y-%m-%d')
        ))
        current = period_end + timedelta(days=1)
    return periods[:36]  # exactement 36 périodes

PERIODS = generate_10day_periods(2021)
print(f"✅ {len(PERIODS)} périodes de 10 jours générées.")
print(f"   Première : {PERIODS[0]}, Dernière : {PERIODS[-1]}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. FONCTIONS DE TRAITEMENT SENTINEL-2
# ─────────────────────────────────────────────────────────────────────────────

def mask_s2_clouds(image):
    """
    Masque les nuages à partir du Quality Assessment band (QA60).
    Supprime les pixels nuageux et les cirrus.
    """
    qa = image.select('QA60')
    cloud_bit_mask  = 1 << 10  # bit 10 = nuages opaques
    cirrus_bit_mask = 1 << 11  # bit 11 = cirrus
    mask = (qa.bitwiseAnd(cloud_bit_mask).eq(0)
              .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0)))
    return image.updateMask(mask).divide(10000)  # conversion réflectance [0,1]


def get_s2_median_image(region, start_date, end_date):
    """
    Récupère l'image médiane Sentinel-2 Level-2A sur une période donnée.
    Les pixels masqués (nuages) sont exclus du calcul de la médiane.
    Retourne une image multi-bandes ou None si aucune image disponible.
    """
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(region)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
                  .map(mask_s2_clouds)
                  .select(S2_BANDS))

    count = collection.size().getInfo()
    if count == 0:
        return None  # données manquantes → marquées 0 dans le CSV
    return collection.median()


def build_temporal_stack(region, periods, area_name):
    """
    Construit une image multi-temporelle avec 10 bandes × 36 périodes = 360 bandes.
    Les périodes sans données sont remplies avec des zéros (conformément à l'article).

    Args:
        region: ee.Geometry — zone d'étude
        periods: list of tuples — 36 intervalles de 10 jours
        area_name: str — nom pour les logs

    Returns:
        ee.Image — stack de 360 bandes nommées [Band_t1, ..., Band_t36]
    """
    print(f"\n📡 Construction du stack temporel pour {area_name}...")
    images = []

    for i, (start, end) in enumerate(periods):
        print(f"   Période {i+1:02d}/36 : {start} → {end}", end='\r')
        img = get_s2_median_image(region, start, end)

        if img is None:
            # Données manquantes → image constante = 0 (convention de l'article)
            zero_bands = [f"{b}_t{i+1}" for b in S2_BANDS]
            img = ee.Image.constant(0).rename('B2').addBands(
                [ee.Image.constant(0).rename(b) for b in S2_BANDS[1:]]
            ).select(S2_BANDS)

        # Renommer les bandes : B2_t1, B3_t1, ..., B12_t36
        new_names = [f"{b}_t{i+1}" for b in S2_BANDS]
        img = img.rename(new_names)
        images.append(img)

    # Empiler toutes les images en une seule image multi-bandes
    stack = ee.Image.cat(images)
    print(f"\n   ✅ Stack créé : {len(images) * len(S2_BANDS)} bandes total")
    return stack


# ─────────────────────────────────────────────────────────────────────────────
# 5. CHARGEMENT DES DONNÉES DE RÉFÉRENCE (CDL + ESA WorldCover)
#    Conformément aux Sections 2.2.1 et 2.2.2 de l'article
# ─────────────────────────────────────────────────────────────────────────────

def load_cdl_2021(region):
    """
    Charge le Cropland Data Layer (CDL) 2021 du USDA.
    Filtre la couche de confiance à 95% (comme dans l'article).
    """
    cdl = ee.ImageCollection('USDA/NASS/CDL').filterDate('2021-01-01', '2021-12-31').first()
    crop_layer      = cdl.select('cropland')
    confidence_layer = cdl.select('confidence')
    # Masque de confiance ≥ 95 % (Section 2.2.4)
    high_confidence_mask = confidence_layer.gte(95)
    return crop_layer.updateMask(high_confidence_mask).clip(region)


def load_esa_worldcover(region):
    """
    Charge ESA WorldCover 2021 pour masquer les zones non-cultivées.
    Classe 40 = terres cultivées (Cropland).
    """
    worldcover = ee.ImageCollection('ESA/WorldCover/v200').first()
    cropland_mask = worldcover.eq(40)  # code 40 = Cropland
    return cropland_mask.clip(region)


def remap_cdl_to_classes(cdl_image, state_name):
    """
    Regroupe les codes CDL en classes d'intérêt selon l'article (Table 2).
    Les cultures < 5% sont fusionnées en 'Others' (code 99).

    Codes CDL USDA principaux :
      1  = Corn         2  = Cotton      3  = Rice
      5  = Soybeans     36 = Alfalfa     69 = Grapes
      75 = Almonds      204 = Pistachios
    """
    if state_name == 'Arkansas':
        # CDL → classe MCTNet
        from_codes = ee.List([1, 2, 3, 5])
        to_codes   = ee.List([3, 4, 2, 1])  # Corn=3, Cotton=4, Rice=2, Soybeans=1
        remapped = cdl_image.remap(from_codes, to_codes, defaultValue=5)  # 5=Others
    else:  # California
        from_codes = ee.List([69, 3, 36, 75, 204])
        to_codes   = ee.List([1,  2,  3,  4,  5])   # Grapes=1, Rice=2, Alfalfa=3, Almonds=4, Pistachios=5
        remapped = cdl_image.remap(from_codes, to_codes, defaultValue=6)  # 6=Others
    return remapped


# ─────────────────────────────────────────────────────────────────────────────
# 6. ÉCHANTILLONNAGE DES PIXELS (10 000 points par zone)
#    Conformément à la Section 2.2.4 de l'article
# ─────────────────────────────────────────────────────────────────────────────

def sample_pixels(temporal_stack, cdl_labeled, cropland_mask, region,
                  n_samples=10000, seed=42):
    """
    Échantillonne aléatoirement 10 000 pixels dans les zones cultivées.
    Extrait les 360 features spectrales + le type de culture.

    Args:
        temporal_stack: ee.Image — stack 360 bandes spectrales
        cdl_labeled:    ee.Image — labels de culture (remaped)
        cropland_mask:  ee.Image — masque zones cultivées (ESA WorldCover)
        region:         ee.Geometry — zone d'étude
        n_samples:      int — nombre de pixels à échantillonner
        seed:           int — graine aléatoire pour reproductibilité

    Returns:
        ee.FeatureCollection — collection de points avec features
    """
    # Appliquer le masque ESA WorldCover sur le stack
    masked_stack = temporal_stack.updateMask(cropland_mask)
    # Ajouter la couche de labels
    combined = masked_stack.addBands(cdl_labeled.rename('crop_type'))

    # Échantillonnage stratifié aléatoire
    samples = combined.stratifiedSample(
        numPoints=n_samples,
        classBand='crop_type',
        region=region,
        scale=10,            # résolution 10m de Sentinel-2
        seed=seed,
        geometries=True      # inclure les coordonnées lat/lon
    )
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 7. EXPORT VERS GOOGLE DRIVE
# ─────────────────────────────────────────────────────────────────────────────

def export_to_drive(feature_collection, filename, folder='MCTNet_CropMapping'):
    """
    Lance une tâche d'export GEE vers Google Drive au format CSV.
    La tâche s'exécute côté serveur GEE ; surveiller sur code.earthengine.google.com.

    Args:
        feature_collection: ee.FeatureCollection — données à exporter
        filename:           str — nom du fichier CSV (sans extension)
        folder:             str — dossier Google Drive de destination

    Returns:
        ee.batch.Task — tâche GEE lancée
    """
    task = ee.batch.Export.table.toDrive(
        collection=feature_collection,
        description=filename,
        folder=folder,
        fileNamePrefix=filename,
        fileFormat='CSV',
        selectors=['crop_type', 'longitude', 'latitude'] +
                  [f"{b}_t{t}" for t in range(1, 37) for b in S2_BANDS]
    )
    task.start()
    print(f"   🚀 Export lancé : '{filename}.csv' → Google Drive/{folder}/")
    print(f"   🔗 Suivre : https://code.earthengine.google.com/tasks")
    return task


# ─────────────────────────────────────────────────────────────────────────────
# 8. VISUALISATION DES ZONES D'ÉTUDE
# ─────────────────────────────────────────────────────────────────────────────

def visualize_study_areas():
    """
    Affiche une carte matplotlib des zones d'étude aux États-Unis,
    similaire à la Figure 1 de l'article MCTNet.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.set_facecolor('#E8F4FD')
    fig.patch.set_facecolor('#FAFAFA')

    # Contours simplifiés des États-Unis
    us_outline = plt.Polygon([
        [-124.7, 24.5], [-66.9, 24.5], [-66.9, 49.4], [-124.7, 49.4]
    ], fill=True, facecolor='#F5F0E8', edgecolor='#999999', linewidth=1.5)
    ax.add_patch(us_outline)

    # ── Arkansas ──────────────────────────────────────────────────────────────
    ar_rect = plt.Rectangle((-94.62, 33.00), 4.98, 3.50,
                             linewidth=2, edgecolor='#E74C3C',
                             facecolor='#FADBD8', alpha=0.7, label='Arkansas')
    ax.add_patch(ar_rect)
    ax.text(-92.1, 34.75, 'Arkansas', fontsize=11, fontweight='bold',
            color='#C0392B', ha='center', va='center')

    # Mapping areas Arkansas
    ar_ma1 = plt.Rectangle((-92.5, 35.0), 1.0, 1.0,
                             linewidth=1.5, edgecolor='#C0392B',
                             facecolor='#E74C3C', alpha=0.5)
    ar_ma2 = plt.Rectangle((-91.5, 33.5), 1.0, 1.0,
                             linewidth=1.5, edgecolor='#C0392B',
                             facecolor='#E74C3C', alpha=0.5)
    ax.add_patch(ar_ma1)
    ax.add_patch(ar_ma2)
    ax.text(-92.0, 35.5, '1', fontsize=9, color='white', ha='center',
            va='center', fontweight='bold')
    ax.text(-91.0, 34.0, '2', fontsize=9, color='white', ha='center',
            va='center', fontweight='bold')

    # ── Californie ────────────────────────────────────────────────────────────
    ca_rect = plt.Rectangle((-124.48, 32.53), 10.35, 9.48,
                             linewidth=2, edgecolor='#2980B9',
                             facecolor='#D6EAF8', alpha=0.7, label='California')
    ax.add_patch(ca_rect)
    ax.text(-119.3, 37.0, 'California', fontsize=11, fontweight='bold',
            color='#1A5276', ha='center', va='center')

    # Mapping areas California
    ca_ma1 = plt.Rectangle((-122.5, 38.5), 1.0, 1.0,
                             linewidth=1.5, edgecolor='#1A5276',
                             facecolor='#2980B9', alpha=0.5)
    ca_ma2 = plt.Rectangle((-120.5, 36.5), 1.0, 1.0,
                             linewidth=1.5, edgecolor='#1A5276',
                             facecolor='#2980B9', alpha=0.5)
    ax.add_patch(ca_ma1)
    ax.add_patch(ca_ma2)
    ax.text(-122.0, 39.0, '1', fontsize=9, color='white', ha='center',
            va='center', fontweight='bold')
    ax.text(-120.0, 37.0, '2', fontsize=9, color='white', ha='center',
            va='center', fontweight='bold')

    # Mise en forme
    ax.set_xlim(-127, -64)
    ax.set_ylim(23, 52)
    ax.set_xlabel('Longitude (°)', fontsize=12)
    ax.set_ylabel('Latitude (°)', fontsize=12)
    ax.set_title('Zones d\'étude — Cartographie des cultures (MCTNet, 2021)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Légende
    legend_elements = [
        mpatches.Patch(facecolor='#FADBD8', edgecolor='#E74C3C', label='Arkansas'),
        mpatches.Patch(facecolor='#D6EAF8', edgecolor='#2980B9', label='California'),
        mpatches.Patch(facecolor='#E74C3C', alpha=0.5, label='Mapping areas'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10,
              framealpha=0.9)

    # Annotation
    ax.text(-75, 26, 'États-Unis', fontsize=9, color='#666666',
            style='italic', alpha=0.7)

    plt.tight_layout()
    plt.savefig('study_areas_map.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Carte sauvegardée : study_areas_map.png")


def visualize_temporal_periods():
    """
    Visualise les 36 périodes de 10 jours de l'année 2021 sur un axe temporel.
    """
    fig, ax = plt.subplots(figsize=(14, 4))

    colors = ['#3498DB' if i % 3 != 2 else '#E74C3C' for i in range(36)]
    for i, (start, end) in enumerate(PERIODS):
        doy_start = datetime.strptime(start, '%Y-%m-%d').timetuple().tm_yday
        ax.barh(0, 10, left=doy_start, height=0.6,
                color=colors[i], alpha=0.8, edgecolor='white', linewidth=0.5)
        if i % 6 == 0:
            ax.text(doy_start + 5, 0, f't{i+1}', ha='center', va='center',
                    fontsize=7, color='white', fontweight='bold')

    # Mois
    months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun',
              'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    for doy, month in zip(month_starts, months):
        ax.axvline(x=doy, color='#666666', linestyle='--', alpha=0.4, linewidth=0.8)
        ax.text(doy + 15, -0.55, month, ha='center', fontsize=9, color='#444444')

    ax.set_xlim(1, 365)
    ax.set_ylim(-0.8, 0.8)
    ax.set_xlabel('Jour de l\'année (DOY)', fontsize=11)
    ax.set_title('36 Périodes temporelles Sentinel-2 — Année 2021', fontsize=13,
                 fontweight='bold')
    ax.set_yticks([])

    # Légende
    legend_elements = [
        mpatches.Patch(facecolor='#3498DB', label='Périodes 1-2 du mois'),
        mpatches.Patch(facecolor='#E74C3C', label='3ème période du mois'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    ax.grid(axis='x', alpha=0.2)

    plt.tight_layout()
    plt.savefig('temporal_periods.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Diagramme temporel sauvegardé : temporal_periods.png")


# ─────────────────────────────────────────────────────────────────────────────
# 9. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(state_name, config, n_samples=10000, drive_folder='MCTNet_CropMapping'):
    """
    Pipeline complet pour une zone d'étude :
      1. Charge CDL 2021 + ESA WorldCover
      2. Construit le stack temporel Sentinel-2 (360 bandes)
      3. Échantillonne 10 000 pixels
      4. Exporte vers Google Drive

    Args:
        state_name:    str — 'Arkansas' ou 'California'
        config:        dict — configuration de la zone (bbox, classes CDL)
        n_samples:     int — nombre de pixels à échantillonner
        drive_folder:  str — dossier Google Drive de destination
    """
    print(f"\n{'='*60}")
    print(f"  TRAITEMENT : {state_name.upper()}")
    print(f"{'='*60}")

    region = config['bbox']

    # Étape 1 : Données de référence
    print("\n📋 Chargement des données de référence...")
    cdl_raw     = load_cdl_2021(region)
    cdl_labeled = remap_cdl_to_classes(cdl_raw, state_name)
    cropland_mask = load_esa_worldcover(region)
    print("   ✅ CDL 2021 + ESA WorldCover chargés")

    # Étape 2 : Stack temporel Sentinel-2
    temporal_stack = build_temporal_stack(region, PERIODS, state_name)

    # Étape 3 : Échantillonnage
    print(f"\n🎯 Échantillonnage de {n_samples} pixels...")
    samples = sample_pixels(
        temporal_stack, cdl_labeled, cropland_mask,
        region, n_samples=n_samples
    )
    print(f"   ✅ Échantillonnage configuré")

    # Étape 4 : Export vers Google Drive
    print(f"\n💾 Export vers Google Drive...")
    filename = f"sentinel2_crops_{state_name.lower()}_2021"
    task = export_to_drive(samples, filename, folder=drive_folder)

    return task


# ─────────────────────────────────────────────────────────────────────────────
# 10. POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ── Visualisations préliminaires ─────────────────────────────────────────
    print("\n📊 Génération des visualisations...")
    visualize_study_areas()
    visualize_temporal_periods()

    # ── Lancement du pipeline pour chaque zone ───────────────────────────────
    tasks = {}
    for state_name, config in STUDY_AREAS.items():
        task = run_pipeline(
            state_name=state_name,
            config=config,
            n_samples=10000,
            drive_folder='MCTNet_CropMapping_2021'
        )
        tasks[state_name] = task

    # ── Résumé final ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  RÉSUMÉ DES TÂCHES D'EXPORT GEE")
    print(f"{'='*60}")
    for state, task in tasks.items():
        status = task.status()
        print(f"  {state:12s} : {status['state']} — {status['description']}")

    print("""
╔══════════════════════════════════════════════════════════════╗
║  PROCHAINES ÉTAPES                                           ║
║                                                              ║
║  1. Surveiller les tâches :                                  ║
║     https://code.earthengine.google.com/tasks                ║
║                                                              ║
║  2. Une fois terminé, vos CSVs seront dans :                 ║
║     Google Drive > MCTNet_CropMapping_2021 >                 ║
║       • sentinel2_crops_arkansas_2021.csv                    ║
║       • sentinel2_crops_california_2021.csv                  ║
║                                                              ║
║  3. Structure de chaque CSV :                                ║
║     [crop_type, latitude, longitude,                         ║
║      B2_t1, B3_t1, ..., B12_t1,  ← période 1                ║
║      B2_t2, B3_t2, ..., B12_t2,  ← période 2                ║
║      ...                                                     ║
║      B2_t36, ..., B12_t36]       ← période 36               ║
║     → 360 features par pixel (10 bandes × 36 périodes)       ║
╚══════════════════════════════════════════════════════════════╝
    """)


# ─────────────────────────────────────────────────────────────────────────────
# ANNEXE : Vérification rapide des données téléchargées
# ─────────────────────────────────────────────────────────────────────────────

def inspect_csv(csv_path, state_name):
    """
    Charge et inspecte un CSV exporté. À lancer après le téléchargement.

    Usage :
        inspect_csv('/content/drive/MyDrive/MCTNet_CropMapping_2021/sentinel2_crops_arkansas_2021.csv',
                    'Arkansas')
    """
    df = pd.read_csv(csv_path)

    print(f"\n📊 Inspection du CSV — {state_name}")
    print(f"   Forme : {df.shape} ({df.shape[0]} pixels × {df.shape[1]} colonnes)")
    print(f"\n   Distribution des cultures :")
    crop_map = {1:'Soybeans/Grapes', 2:'Rice', 3:'Corn/Alfalfa',
                4:'Cotton/Almonds', 5:'Others/Pistachios', 6:'Others'}
    counts = df['crop_type'].value_counts().sort_index()
    for code, count in counts.items():
        pct = 100 * count / len(df)
        label = crop_map.get(code, f'Code {code}')
        print(f"     {label:20s} : {count:5d} ({pct:.1f}%)")

    # Vérification des valeurs manquantes
    feature_cols = [c for c in df.columns if c.startswith('B')]
    missing_rate = (df[feature_cols] == 0).mean().mean() * 100
    print(f"\n   Taux de valeurs manquantes (pixels = 0) : {missing_rate:.1f}%")

    # Visualisation NDVI moyen par culture
    if 'B8_t1' in df.columns and 'B4_t1' in df.columns:
        print(f"\n   🌿 Calcul des profils NDVI moyens...")
        fig, ax = plt.subplots(figsize=(12, 5))
        doys = [10 + i*10 for i in range(36)]

        for crop_code in sorted(df['crop_type'].unique()):
            subset = df[df['crop_type'] == crop_code]
            ndvi_profile = []
            for t in range(1, 37):
                nir  = subset[f'B8_t{t}'].replace(0, np.nan)
                red  = subset[f'B4_t{t}'].replace(0, np.nan)
                ndvi = ((nir - red) / (nir + red + 1e-10)).mean()
                ndvi_profile.append(ndvi)
            label = crop_map.get(crop_code, f'Code {crop_code}')
            ax.plot(doys, ndvi_profile, marker='o', markersize=4,
                    linewidth=1.5, label=label)

        ax.set_xlabel('Jour de l\'année (DOY)', fontsize=11)
        ax.set_ylabel('NDVI moyen', fontsize=11)
        ax.set_title(f'Profils NDVI temporels — {state_name} 2021', fontsize=13,
                     fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'ndvi_profiles_{state_name.lower()}.png', dpi=150)
        plt.show()
        print(f"   ✅ Profils NDVI sauvegardés")

    return df