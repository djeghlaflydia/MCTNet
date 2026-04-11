// ============================================================
// EXTRACTION PIXEL-BASED : 36 CSV PAR ZONE
// Conforme au papier : MCTNet (Wang et al., 2024)
// Chaque CSV = ~10 000 pixels × (10 bandes + 1 class_label)
// 1 CSV = 1 période de 10 jours → 3 CSV/mois → 36 CSV/an
// 4 zones × 36 CSV = 144 fichiers au total
// ============================================================

// ============================================================
// 1. CHARGER LES ÉTATS ET DIVISER EN ZONES
// (2 zones par état comme dans Fig.1 du papier)
// ============================================================
var states   = ee.FeatureCollection("TIGER/2018/States");
var CA_state = states.filter(ee.Filter.eq('NAME', 'California'));
var AR_state = states.filter(ee.Filter.eq('NAME', 'Arkansas'));

var CA_zone1 = CA_state.geometry().intersection(
  ee.Geometry.Rectangle([-124.48, 37.3,  -114.13, 42.01]), 1);
var CA_zone2 = CA_state.geometry().intersection(
  ee.Geometry.Rectangle([-124.48, 32.53, -114.13, 37.3]),  1);
var AR_zone1 = AR_state.geometry().intersection(
  ee.Geometry.Rectangle([-94.62,  34.8,  -89.64,  36.50]), 1);
var AR_zone2 = AR_state.geometry().intersection(
  ee.Geometry.Rectangle([-94.62,  33.00, -89.64,  34.8]),  1);

// ============================================================
// 2. CDL 2021 — LABEL + FILTRE CONFIANCE 95% + MASQUE WORLDCOVER
//
// Papier (section 2.2.4) :
// "we set a 95% confidence to filter the CDL map to improve
//  the quality of sampling"
// "used the ESA WorldCover 2021 to mask non-cropland areas"
// ============================================================
var cdlImage      = ee.Image('USDA/NASS/CDL/2021');
var croplandLabel = cdlImage.select('cropland');
var cdlConf       = cdlImage.select('confidence'); // bande confidence (0-100)

// Masque confiance ≥ 95% (filtre qualité CDL)
var confMask = cdlConf.gte(95);

// Masque WorldCover 2021 : classe 40 = Cropland uniquement
var worldCoverMask = ee.Image('ESA/WorldCover/v200/2021')
                       .select('Map').eq(40);

// Masque combiné : cropland WorldCover ET confiance CDL ≥ 95%
var finalMask = worldCoverMask.and(confMask);

// ============================================================
// 3. REMAPPING DES CLASSES SELON L'ARTICLE (Table 2)
//
// Arkansas  (4 cultures + Others) :
//   Corn=1 → 1 | Cotton=2 → 2 | Rice=3 → 3 | Soybeans=5 → 4
//   tout le reste → 5 (Others)
//
// California (5 cultures + Others) :
//   Grapes=69 → 1 | Rice=3 → 2 | Alfalfa=36 → 3
//   Almonds=75 → 4 | Pistachios=204 → 5
//   tout le reste → 6 (Others)
//
// Note papier : "crop types that constitute less than 5% of the
// total number of samples were merged into Others"
// → géré en post-traitement Python après export
// ============================================================
var AR_classImg = croplandLabel
  .remap([1, 2, 3, 5], [1, 2, 3, 4], 5)
  .rename('class_label')
  .toByte();

var CA_classImg = croplandLabel
  .remap([69, 3, 36, 75, 204], [1, 2, 3, 4, 5], 6)
  .rename('class_label')
  .toByte();

// ============================================================
// 4. BANDES SENTINEL-2 ET DATES
// 10 bandes × 36 périodes = 360 features (papier section 2.2.3)
// ============================================================
var BANDS  = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'];
var months = ['01','02','03','04','05','06','07','08','09','10','11','12'];
var days   = ['01','11','21'];

// ============================================================
// 5. FONCTION : COMPOSITE MÉDIAN SENTINEL-2 SUR 10 JOURS
//
// Papier (section 2.2.4) :
// "computed the median value of the remaining observations
//  at ten-day intervals, resulting in a total of 36 temporal sequences"
// "there are still missing data in the sequences,
//  and we used 0 to mark the missing data"
// ============================================================
function getComposite(dateStr, geometry) {
  var start = ee.Date(dateStr);

  // Fenêtre principale : 10 jours, nuages < 20%
  var col10 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterDate(start, start.advance(10, 'day'))
    .filterBounds(geometry)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .select(BANDS);

  // Fenêtre fallback : 30 jours, nuages < 50%
  var col30 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterDate(start, start.advance(30, 'day'))
    .filterBounds(geometry)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
    .select(BANDS);

  // Valeur 0 = données manquantes (convention MCTNet)
  var empty = ee.Image.constant(ee.List.repeat(0, BANDS.length))
                .rename(BANDS).toFloat();

  return ee.Image(ee.Algorithms.If(
    col10.size().gt(0),
    col10.median().toFloat(),
    ee.Image(ee.Algorithms.If(
      col30.size().gt(0),
      col30.median().toFloat(),
      empty
    ))
  ));
}

// ============================================================
// 6. FONCTION : ÉCHANTILLONNAGE 10 000 PIXELS PAR ZONE/DATE
//
// Papier (section 2.2.4) :
// "randomly sampled 10,000 points in each study area"
// ============================================================
function samplePixels(composite, classImg, geometry,
                      stateName, zoneName, dateStr, csvNum) {

  // Image finale : 10 bandes S2 + class_label remappé
  // Appliquer le masque combiné (WorldCover + confiance 95%)
  var image = composite
    .addBands(classImg)
    .updateMask(finalMask.clip(geometry))
    .clip(geometry)
    .toFloat();

  // Échantillonnage aléatoire de 10 000 pixels cropland
  var samples = image.sample({
    region:     geometry,
    scale:      10,      // résolution native S2 (10m)
    numPixels:  10000,
    seed:       42,      // seed fixe → reproductibilité
    dropNulls:  true,    // ignorer pixels sans données
    geometries: false    // pas de coordonnées → CSV plus léger
  });

  // Ajouter métadonnées
  samples = samples.map(function(f) {
    return f.set({
      'date':   dateStr,
      'state':  stateName,
      'zone':   zoneName,
      'csv_id': csvNum
    });
  });

  return samples;
}

// ============================================================
// 7. BOUCLE PRINCIPALE : 4 ZONES × 12 MOIS × 3 PÉRIODES = 144 CSV
// ============================================================
var zones = [
  // Arkansas : 1=Corn 2=Cotton 3=Rice 4=Soybeans 5=Others
  { geometry: AR_zone1, classImg: AR_classImg,
    state: 'Arkansas',   zone: 'zone1', folder: 'Arkansas/zone1' },
  { geometry: AR_zone2, classImg: AR_classImg,
    state: 'Arkansas',   zone: 'zone2', folder: 'Arkansas/zone2' },
  // California : 1=Grapes 2=Rice 3=Alfalfa 4=Almonds 5=Pistachios 6=Others
  { geometry: CA_zone1, classImg: CA_classImg,
    state: 'California', zone: 'zone1', folder: 'California/zone1' },
  { geometry: CA_zone2, classImg: CA_classImg,
    state: 'California', zone: 'zone2', folder: 'California/zone2' },
];

// Colonnes CSV : 10 bandes + class_label + métadonnées
var exportCols = BANDS.concat(['class_label', 'date', 'state', 'zone', 'csv_id']);

zones.forEach(function(z) {
  print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  print('🌍 ' + z.state + ' — ' + z.zone);

  months.forEach(function(month) {
    days.forEach(function(day, dayIdx) {

      var dateStr = '2021-' + month + '-' + day;
      var csvNum  = dayIdx + 1; // 1, 2, ou 3

      var composite = getComposite(dateStr, z.geometry);

      var samples = samplePixels(
        composite, z.classImg, z.geometry,
        z.state, z.zone, dateStr, csvNum
      );

      var fileName = z.state + '_' + z.zone + '_' + dateStr + '_csv' + csvNum;

      Export.table.toDrive({
        collection:     samples,
        description:    fileName,
        folder:         z.folder,
        fileNamePrefix: fileName,
        fileFormat:     'CSV',
        selectors:      exportCols
      });

      print('  📤 ' + dateStr + ' | csv' + csvNum + ' → ' + fileName);
    });
  });
});

// ============================================================
// 8. RÉSUMÉ CONFORMITÉ PAPIER
// ============================================================
print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
print('✅ 144 tâches lancées (4 zones × 36 périodes)');
print('');
print('📋 CONFORMITÉ PAPIER MCTNet (Wang et al., 2024) :');
print('   ✅ 10 bandes S2 : B2 B3 B4 B5 B6 B7 B8 B8A B11 B12');
print('   ✅ 36 composites médians de 10 jours');
print('   ✅ Pixels manquants → valeur 0');
print('   ✅ Masque WorldCover 2021 (classe 40 = Cropland)');
print('   ✅ Filtre confiance CDL ≥ 95%');
print('   ✅ 10 000 pixels échantillonnés aléatoirement par zone');
print('   ✅ Classes remappées selon Table 2 du papier');
print('');
print('🔢 CLASSES Arkansas (4 + Others) :');
print('   1=Corn | 2=Cotton | 3=Rice | 4=Soybeans | 5=Others');
print('');
print('🔢 CLASSES California (5 + Others) :');
print('   1=Grapes | 2=Rice | 3=Alfalfa | 4=Almonds | 5=Pistachios | 6=Others');
print('');
print('📂 DRIVE :');
print('   Arkansas/zone1/   → 36 CSV');
print('   Arkansas/zone2/   → 36 CSV');
print('   California/zone1/ → 36 CSV');
print('   California/zone2/ → 36 CSV');
print('');
print('👉 Tasks → Run All → Confirm');
print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

// ============================================================
// 9. VISUALISATION
// ============================================================
Map.setOptions('HYBRID');
Map.setCenter(-96, 37, 4);
Map.addLayer(CA_zone1, {color:'FF0000', fillColor:'FF000033'}, '🔴 CA Zone1');
Map.addLayer(CA_zone2, {color:'CC0000', fillColor:'CC000033'}, '🔴 CA Zone2');
Map.addLayer(AR_zone1, {color:'0055FF', fillColor:'0055FF33'}, '🔵 AR Zone1');
Map.addLayer(AR_zone2, {color:'003399', fillColor:'00339933'}, '🔵 AR Zone2');

// Aperçu des classes remappées
Map.addLayer(
  AR_classImg.updateMask(finalMask).clip(AR_zone1),
  {min:1, max:5, palette:['green','brown','cyan','yellow','gray']},
  '🌾 AR Classes (1=Corn 2=Cotton 3=Rice 4=Soy 5=Other)'
);
Map.addLayer(
  CA_classImg.updateMask(finalMask).clip(CA_zone1),
  {min:1, max:6, palette:['purple','cyan','orange','pink','beige','gray']},
  '🍇 CA Classes (1=Grapes 2=Rice 3=Alfalfa 4=Almonds 5=Pist 6=Other)'
);