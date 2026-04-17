// ================================================================
// MCTNet — Part 2: Environmental Covariates Extraction
// Project: Deep Learning for Crop Classification (USTHB)
//
// This script extracts Climate, Soil, and Topography covariates
// for the same points used in the baseline Sentinel-2 dataset.
// ================================================================

// ── ★ CONFIGURATION (Must match extracteData.js) ★ ────────────
var ZONE = 'arkansas';   // 'arkansas' or 'california'
var YEAR = 2021;
var SEED = 42;

var ZONES = {
  arkansas: {
    region: ee.Geometry.Rectangle([-94.62, 33.00, -89.64, 36.50]),
    exactCounts: [616, 1522, 762, 2423, 4677],
    nClasses: 4
  },
  california: {
    region: ee.Geometry.Rectangle([-122.50, 35.00, -117.50, 40.50]),
    exactCounts: [3512, 2054, 2037, 974, 783, 640],
    nClasses: 5
  }
};

var cfg    = ZONES[ZONE];
var region = cfg.region;

// ================================================================
// 1. REPRODUCE SAMPLING POINTS
// ================================================================
var cdl = ee.Image('USDA/NASS/CDL/2021').select('cropland');
var labelBase = ee.Image(0).byte().rename('label');
var CODES = (ZONE === 'arkansas') ? [1, 2, 3, 5] : [69, 3, 36, 75, 204];

for (var k = 0; k < cfg.nClasses; k++) {
  labelBase = labelBase.where(cdl.eq(CODES[k]), k + 1);
}

var labelImg = labelBase.addBands(ee.Image.pixelLonLat());

var cVals = [];
for (var i = 0; i <= cfg.nClasses; i++) { cVals.push(i); }

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
  var lon = f.get('longitude');
  var lat = f.get('latitude');
  return f.set('pixel_id', f.get('system:index'))
          .setGeometry(ee.Geometry.Point([lon, lat]));
});

// ================================================================
// 2. COVARIATE DATASETS
// ================================================================

// A. TOPOGRAPHY (Static)
var srtm = ee.Image('USGS/SRTMGL1_003');
var elevation = srtm.select('elevation');
var slope     = ee.Terrain.slope(elevation).rename('slope');
var aspect    = ee.Terrain.aspect(elevation).rename('aspect');
var topoImg   = elevation.addBands(slope).addBands(aspect);

// B. SOIL (Static - OpenLandMap 0-20cm mean)
var clay = ee.Image('OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02').select('b0').rename('clay');
var sand = ee.Image('OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02').select('b0').rename('sand');
var oc   = ee.Image('OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02').select('b0').rename('org_carbon');
var ph   = ee.Image('OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02').select('b0').rename('ph');
var soilImg = clay.addBands(sand).addBands(oc).addBands(ph);

// C. CLIMATE (Temporal - ERA5-Land Monthly)
var era5 = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
             .filterDate(YEAR + '-01-01', YEAR + '-12-31')
             .select(['temperature_2m', 'total_precipitation_sum'], ['temp', 'precip']);

// ================================================================
// 3. EXTRACTION
// ================================================================

// Static Covariates
var staticCovariates = topoImg.addBands(soilImg);
var staticData = staticCovariates.sampleRegions({
  collection: allPoints,
  scale: 30, // Resolution of SRTM
  geometries: false
});

// Temporal Covariates (ERA5)
// We will extract climate for each of the 12 months
var climateData = era5.map(function(img) {
  var month = ee.Date(img.get('system:time_start')).get('month');
  return img.sampleRegions({
    collection: allPoints,
    scale: 1000, // ERA5 resolution is coarse
    geometries: false
  }).map(function(f) {
    return f.set('month', month);
  });
}).flatten();

// ================================================================
// 4. EXPORT
// ================================================================

Export.table.toDrive({
  collection: staticData,
  description: 'covariates_static_' + ZONE,
  folder: 'MCTNet_' + ZONE,
  fileNamePrefix: 'covariates_static_' + ZONE,
  fileFormat: 'CSV'
});

Export.table.toDrive({
  collection: climateData,
  description: 'covariates_climate_' + ZONE,
  folder: 'MCTNet_' + ZONE,
  fileNamePrefix: 'covariates_climate_' + ZONE,
  fileFormat: 'CSV'
});

print('★ Export tasks created for Zone: ' + ZONE);
print('  Please run both tasks in the Tasks panel.');
