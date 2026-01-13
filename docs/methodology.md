# Methodology

## Overview
The proposed system follows a multi-stage remote sensing workflow designed for detecting forest cover loss in non-tropical regions. The methodology integrates optical and radar satellite data to overcome limitations caused by cloud cover and seasonal snow.

## Step 1: Study Area Selection
A representative non-tropical region (temperate or boreal forest) is selected for analysis. The region is defined spatially using geographic boundaries within Google Earth Engine.

## Step 2: Data Acquisition
Multi-temporal Sentinel-2 optical imagery and Sentinel-1 SAR imagery are accessed through Google Earth Engine. Data is filtered based on date range, cloud cover (for optical data), and polarization modes (for SAR data).

## Step 3: Preprocessing
- Sentinel-2 imagery is preprocessed using cloud masking and vegetation index computation.
- Sentinel-1 SAR data is preprocessed using SNAP for radiometric calibration, speckle filtering, and terrain correction.
Preprocessed data is prepared for time-series analysis.

## Step 4: Change Detection
Forest cover changes are detected by comparing vegetation and backscatter metrics across different time periods. Significant reductions in vegetation indices or radar backscatter are flagged as potential deforestation events.

## Step 5: Validation
Detected forest loss areas are cross-validated using Global Forest Watch forest loss datasets to reduce false positives.

## Step 6: Visualization
Results are visualized as thematic maps highlighting forest loss locations and temporal patterns. These outputs support interpretation and decision-making.
