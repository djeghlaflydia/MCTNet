// ================================================================
// MCTNet — Part 2: Environmental Covariates Extraction
// Project: Deep Learning for Crop Classification (USTHB)
//
// This script extracts Climate, Soil, and Topography covariates
// for the same points used in the baseline Sentinel-2 dataset.
//
// ── CHANGES vs. original ──────────────────────────────────────
//  TOPOGRAPHY : elevation + landforms (CSP grFEE)
//               slope & aspect REMOVED
//  SOIL       : clay + org_carbon + ph
//               sand REMOVED
//  CLIMATE    : temp_2m + total_precipitation + solar_radiation
//               (ERA5-Land Monthly, 3 variables)
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

// ── A. TOPOGRAPHY (Static) ────────────────────────────────────
// Variables kept  : elevation, landforms
// Variables removed: slope, aspect
//
// Landforms are derived from the CSP gRFEE (Geomorpho90m) dataset.
// The 'landforms' band encodes 15 geomorphological classes
// (plains, valleys, ridges, peaks, etc.) following the
// Iwahashi & Pike (2007) scheme — directly relevant to soil
// water retention and micro-climate that drive crop growth.
var srtm      = ee.Image('USGS/SRTMGL1_003');
var elevation = srtm.select('elevation').rename('elevation');

// CSP grFEE landforms at 90 m resolution (available globally)
var landforms = ee.Image('CSP/ERGo/1_0/Global/ALOS_landforms')
                  .select('constant')
                  .rename('landforms');

var topoImg = elevation.addBands(landforms);

// ── B. SOIL (Static — OpenLandMap 0–20 cm mean) ───────────────
// Variables kept  : clay, org_carbon, ph
// Variable removed: sand  (highly correlated with clay → redundant)
var clay = ee.Image('OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02')
             .select('b0').rename('clay');
var oc   = ee.Image('OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02')
             .select('b0').rename('org_carbon');
var ph   = ee.Image('OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02')
             .select('b0').rename('ph');

var soilImg = clay.addBands(oc).addBands(ph);

// ── C. CLIMATE (Temporal — ERA5-Land Monthly) ─────────────────
// 3 variables selected for crop classification:
//
//  1. temperature_2m        → drives phenological development
//                             (GDD accumulation, frost risk)
//  2. total_precipitation_sum → water availability / irrigation need,
//                             strongly discriminates crop types
//                             (rice vs. dryland crops)
//  3. surface_solar_radiation_downwards_sum (ssrd) → photosynthetically
//                             active energy input; differentiates
//                             crops with distinct canopy architectures
//                             (e.g. tall corn vs. low-growing cotton)
//
// All three are monthly → 12 values per variable per point (36 features total).
var era5 = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
             .filterDate(YEAR + '-01-01', YEAR + '-12-31')
             .select(
               ['temperature_2m',
                'total_precipitation_sum',
                'surface_solar_radiation_downwards_sum'],
               ['temp', 'precip', 'solar_rad']
             );

// ================================================================
// 3. EXTRACTION
// ================================================================

// ── Static Covariates (Topography + Soil) ─────────────────────
// Scale = 30 m to match SRTM native resolution.
// The landforms layer (90 m) will be resampled automatically by GEE.
var staticCovariates = topoImg.addBands(soilImg);

var staticData = staticCovariates.sampleRegions({
  collection: allPoints,
  scale      : 30,
  geometries : false
});

// ── Temporal Covariates (ERA5 — 12 months × 3 variables) ──────
var climateData = era5.map(function(img) {
  var month = ee.Date(img.get('system:time_start')).get('month');
  return img.sampleRegions({
    collection: allPoints,
    scale      : 1000,   // ERA5-Land native resolution
    geometries : false
  }).map(function(f) {
    return f.set('month', month);
  });
}).flatten();

// ================================================================
// 4. EXPORT
// ================================================================

// Static export  →  one row per point
//   columns: pixel_id, label, elevation, landforms,
//            clay, org_carbon, ph
Export.table.toDrive({
  collection    : staticData,
  description   : 'covariates_static_' + ZONE,
  folder        : 'MCTNet_' + ZONE,
  fileNamePrefix: 'covariates_static_' + ZONE,
  fileFormat    : 'CSV'
});

// Climate export  →  12 rows per point (one per month)
//   columns: pixel_id, label, month, temp, precip, solar_rad
Export.table.toDrive({
  collection    : climateData,
  description   : 'covariates_climate_' + ZONE,
  folder        : 'MCTNet_' + ZONE,
  fileNamePrefix: 'covariates_climate_' + ZONE,
  fileFormat    : 'CSV'
});

print('★ Export tasks created for Zone: ' + ZONE);
print('  Static  → elevation, landforms | clay, org_carbon, ph');
print('  Climate → temp, precip, solar_rad  (12 months each)');
print('  Please run both tasks in the Tasks panel.');