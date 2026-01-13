// STEP 5: Sentinel-2 NDVI computation (Non-Tropical Regions)

// Define study area (example: British Columbia, Canada)
var aoi = ee.Geometry.Rectangle([-130, 48, -114, 60]);

Map.centerObject(aoi, 5);
Map.addLayer(aoi, {color: 'red'}, 'Study Area');

// Cloud masking function for Sentinel-2
function maskS2clouds(image) {
  var qa = image.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
              .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(mask).divide(10000);
}

// Load Sentinel-2 surface reflectance data
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi)
  .filterDate('2022-06-01', '2022-09-30')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(maskS2clouds);

// Create median composite
var composite = s2.median();

// Compute NDVI
var ndvi = composite.normalizedDifference(['B8', 'B4']).rename('NDVI');

// Visualization parameters
var ndviVis = {
  min: 0,
  max: 1,
  palette: ['white', 'green']
};

// Add layers to map
Map.addLayer(ndvi, ndviVis, 'NDVI');

// Print outputs for verification
print('Sentinel-2 NDVI image:', ndvi);
