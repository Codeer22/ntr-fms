# System Architecture

## Overview
The system architecture follows a modular, cloud-assisted remote sensing pipeline. Data acquisition, processing, analysis, and visualization are handled through well-defined components to ensure scalability and reproducibility.

## Architecture Components

### 1. Data Sources
- Sentinel-1 SAR imagery (ESA)
- Sentinel-2 optical imagery (ESA)
- Global Forest Watch forest loss datasets

### 2. Data Processing Layer
- Google Earth Engine is used for large-scale data filtering, preprocessing, and multi-temporal analysis.
- SNAP is used for detailed Sentinel-1 SAR preprocessing tasks such as calibration, speckle filtering, and terrain correction.

### 3. Analysis Layer
- Vegetation indices and radar backscatter metrics are analyzed across time periods.
- Change detection logic identifies significant deviations indicative of forest loss.

### 4. Validation Layer
- Detected deforestation events are cross-verified using Global Forest Watch datasets to reduce false positives.

### 5. Visualization Layer
- Results are visualized as spatial maps and time-series plots.
- Outputs are prepared for interpretation and reporting.

## Workflow Summary
1. Satellite data ingestion
2. Preprocessing and noise reduction
3. Multi-temporal change detection
4. Validation using external datasets
5. Visualization and reporting
