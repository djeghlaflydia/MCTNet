// ================================================================
// MCTNet — GEE Data Extraction (Final Corrected Version v12)
// Paper: Wang et al., 2024
// Project: M1 SII USTHB 2025/2026
//
// Fixes:
//   v2 : CDL → ee.Image direct asset ID
//   v3 : WorldCover → plain JS loop
//   v4 : Reapply validMask after where(); pixel_id from system:index
//   v5 : frequencyHistogram → ee.Reducer.count()
//   v6 : ee.Reducer.count() → aggregate_histogram
//   v7 : ee.Dictionary.fromLists → plain JS; ee.List index lookup
//   v8 : reduceRegions() → sampleRegions(); scale 30→100
//   v9 : centroid → ee.ErrorMargin(1); tileScale:4
//   v10: drop WorldCover mask; drop centroid(); scale→250
//   v11: pixelLonLat() baked into composite; geometries:false
//        in sampleRegions
//   v12: PERMANENT centroid fix — bake pixelLonLat into labelImg
//        at sample() time so allPoints carries lon/lat as properties.
//        Reconstruct true ee.Geometry.Point from those properties
//        before sampleRegions — points have no area so centroid
//        is never invoked anywhere in the pipeline.
//        Per-class scale: 50 for crops, 150 for Others.
// ================================================================

// ── ★ CHANGE THIS BEFORE EACH RUN ★ ──────────────────────────
var ZONE = 'california';   // 'arkansas' or 'california'
// ─────────────────────────────────────────────────────────────

var YEAR     = 2021;
var CDL_CONF = 50;
var SEED     = 42;

// ================================================================
// 1. ZONE CONFIGURATION
// ================================================================
var ZONES = {
  arkansas: {
    region         : ee.Geometry.Rectangle([-94.62, 33.00, -89.64, 36.50]),
    cdlCodes       : [1,      2,        3,      5],
    classNames     : ['Corn', 'Cotton', 'Rice', 'Soybeans'],
    nClasses       : 4,
    // Exact paper counts: Others(0), Corn(1), Cotton(2), Rice(3), Soybeans(4)
    exactCounts    : [616, 1522, 762, 2423, 4677],
    samplesPerClass: 2500
  },
  california: {
    region         : ee.Geometry.Rectangle([-122.50, 35.00, -117.50, 40.50]),
    cdlCodes       : [69,       3,      36,        75,        204],
    classNames     : ['Grapes', 'Rice', 'Alfalfa', 'Almonds', 'Pistachios'],
    nClasses       : 5,
    // Exact paper counts: Others(0), Grapes(1), Rice(2), Alfalfa(3), Almonds(4), Pistachios(5)
    exactCounts    : [3512, 2054, 2037, 974, 783, 640],
    samplesPerClass: 2000
  }
};

var cfg    = ZONES[ZONE];
var region = cfg.region;
var CODES  = cfg.cdlCodes;
var NAMES  = cfg.classNames;
var N      = cfg.nClasses;
var SPC    = cfg.samplesPerClass;

var BAND_NAMES = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'];

// ================================================================
// 2. CDL 2021
// ================================================================
var cdl        = ee.Image('USDA/NASS/CDL/2021');
var cropland   = cdl.select('cropland');
var confidence = cdl.select('confidence');

// ================================================================
// 3. QUALITY MASK
// ================================================================
var validMask = confidence.gte(CDL_CONF);

// ================================================================
// 4. LABEL IMAGE  (0=Others, 1..N=target crops)
//
// Bake pixelLonLat in now so sampling picks up lon/lat as
// plain numeric properties — no geometry calls ever needed.
// ================================================================
var labelBase = ee.Image(0).byte().rename('label');
for (var k = 0; k < N; k++) {
  labelBase = labelBase.where(cropland.eq(CODES[k]), k + 1);
}
labelBase = labelBase.updateMask(validMask);

// Attach lon/lat bands to the label image used for sampling
var labelImg = labelBase.addBands(ee.Image.pixelLonLat());
// labelImg bands: ['label', 'longitude', 'latitude']

// ================================================================
// 5. EXACT STRATIFIED SAMPLING (Perfect Table 2 match)
//
// We use stratifiedSample with the exact numbers requested
// by the paper to avoid getting flooded by 'Others' class.
// ================================================================

// Prepare class points list dynamically based on exact counts
var cVals = [];
for (var i = 0; i <= N; i++) {
  cVals.push(i);
}

var allPoints = labelImg.stratifiedSample({
  numPoints  : 0,
  classBand  : 'label',
  region     : region,
  scale      : 50,
  classValues: cVals,
  classPoints: cfg.exactCounts,
  seed       : SEED,
  tileScale  : 4,
  geometries : false,
  dropNulls  : true
}).map(function(f) {
  // labelImg already provides 'label', 'longitude', 'latitude'
  var lon = f.get('longitude');
  var lat = f.get('latitude');
  return f
    .set('pixel_id', f.get('system:index'))
    .setGeometry(ee.Geometry.Point([lon, lat]));
});

print('★ Sampling done. Total points (server-side):', allPoints.size());

// ================================================================
// 6. SANITY CHECK
// ================================================================
allPoints.aggregate_histogram('label').evaluate(function(hist) {
  print('★ SANITY CHECK — sampled points per class:');
  if (!hist) {
    print('  ⚠️  histogram returned null — check region/CDL asset.');
    return;
  }
  for (var key in hist) {
    var idx       = parseInt(key, 10);
    var className = idx === 0 ? 'Others' : NAMES[idx - 1];
    print('  Class ' + key + ' (' + className + '): ' + hist[key] + ' pts');
    if (hist[key] === 0) {
      print('  ⚠️  Class ' + key + ' is empty — check CDL code!');
    }
  }
});

// ================================================================
// 7. SENTINEL-2 CLOUD MASKING
// ================================================================
function maskS2clouds(img) {
  var qa     = img.select('QA60');
  var clouds = qa.bitwiseAnd(1 << 10).neq(0)
                 .or(qa.bitwiseAnd(1 << 11).neq(0));
  return img
    .updateMask(clouds.not())
    .select(BAND_NAMES)
    .copyProperties(img, ['system:time_start']);
}

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
           .filterBounds(region)
           .filterDate(YEAR + '-01-01', YEAR + '-12-31')
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
           .map(maskS2clouds);

// ================================================================
// 8. EXPORT — 36 CSV FILES, ONE PER 10-DAY TIMESTEP
//
// FIX v12: allPoints now carries true Point geometries constructed
//          from lon/lat properties. sampleRegions locates each
//          point directly — no centroid computation ever triggered.
//
//          pixelLonLat() still added to composite so longitude and
//          latitude properties are refreshed per-timestep in case
//          of any projection shift.
//
//          geometries:false in sampleRegions — lon/lat from bands.
// ================================================================
var START     = ee.Date(YEAR + '-01-01');
var labelList = ee.List(['Others'].concat(NAMES));

var exportCols = ['pixel_id', 'label', 'label_name', 'zone',
                  'timestep', 'date_str', 'longitude', 'latitude',
                  'B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12',
                  'valid'];

var lonLat = ee.Image.pixelLonLat();

for (var t = 0; t < 36; t++) {
  (function(timestep) {

    var t0     = START.advance(timestep * 10, 'day');
    var t1     = t0.advance(10, 'day');
    var subset = s2.filterDate(t0, t1);

    var hasPx = subset.select(BAND_NAMES).count()
                      .select('B2')
                      .gt(0)
                      .unmask(0)
                      .rename('valid');

    var median    = subset.median().unmask(0).select(BAND_NAMES);
    var composite = median.addBands(hasPx).addBands(lonLat);

    // allPoints has true Point geometries — centroid never called
    var extracted = composite.sampleRegions({
      collection: allPoints,
      scale     : 10,
      geometries: false   // lon/lat come from lonLat bands
    });

    extracted = extracted.map(function(f) {
      var lbl     = ee.Number(f.get('label')).int();
      return f
        .set('timestep',   timestep)
        .set('date_str',   t0.format('YYYY-MM-dd'))
        .set('label_name', labelList.get(lbl))
        .set('zone',       ZONE);
      // 'longitude' and 'latitude' already set as properties from bands
    });

    var month   = String(Math.floor(timestep / 3) + 1);
    if (month.length === 1) month = '0' + month;
    var daySlot = (timestep % 3) + 1;
    var fname   = ZONE + '_t' + (timestep < 10 ? '0' : '') + timestep
                + '_' + month + '_d' + daySlot;

    Export.table.toDrive({
      collection    : extracted,
      description   : fname,
      folder        : 'MCTNet_' + ZONE,
      fileNamePrefix: fname,
      fileFormat    : 'CSV',
      selectors     : exportCols
    });

  })(t);
}

print('★ 36 export tasks created for: ' + ZONE);
print('  Tasks panel → Run All');
print('  Drive folder: MCTNet_' + ZONE + '/');

// ================================================================
// 9. MAP VISUALIZATION
// ================================================================
Map.setOptions('HYBRID');
// Bypassing GEE geometry internal bugs entirely
if (ZONE === 'arkansas') {
  Map.setCenter(-92.13, 34.75, 7);
} else {
  Map.setCenter(-120.00, 37.75, 7);
}
Map.addLayer(region,
  {color: '0000FF', fillColor: '0000FF22'}, ZONE);

Map.addLayer(labelBase.clip(region),
  {min: 0, max: N,
   palette: ['gray','green','brown','cyan','yellow','purple'].slice(0, N+1)},
  'Crop labels');

var vis = s2.filterDate('2021-07-01', '2021-07-31').median();
Map.addLayer(vis.clip(region),
  {bands: ['B8','B4','B3'], min: 0, max: 3000},
  'S2 July 2021 NIR');