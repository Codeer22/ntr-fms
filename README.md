# SAR–Optical Forest Disturbance Monitoring
## Pan-India Himalayan Region | 2017–2024

**Project:** Low-Cost Remote Sensing Framework for Monitoring Forest Disturbance
and Recovery Across the Pan-India Himalayan Region Using SAR–Optical Satellite
Time Series

### Study Area
Jammu & Kashmir · Ladakh · Himachal Pradesh · Uttarakhand · Sikkim · Arunachal Pradesh

### Datasets
| Dataset | GEE ID | Role |
|---------|--------|------|
| GADM 4.1 | Local zip | AOI boundaries |
| SRTM DEM | `USGS/SRTMGL1_003` | Elevation mask |
| Hansen GFC 2024 | `UMD/hansen/global_forest_change_2024_v1_12` | Forest mask + validation |
| Sentinel-2 SR | `COPERNICUS/S2_SR_HARMONIZED` | Optical composites |
| Sentinel-1 GRD | `COPERNICUS/S1_GRD` | SAR composites |
| ESA WorldCover | `ESA/WorldCover/v200` | Validation forest mask |
| ALOS PALSAR-2 | `JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH` | Independent L-band validation |

### Pipeline
- **GEE Project:** ntr-fms
- **Asset Folder:** `projects/ntr-fms/assets/himalayan_forest4`
- **Detection:** dNBR < −0.15 OR dNDVI < −0.07 OR dNDMI < −0.05 OR dVH < −1.5 dB
  (majority voting — ≥2 sources must agree)
- **Validation:** Hansen GFC (primary) + ALOS PALSAR-2 L-band (independent)

### How to Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Authenticate GEE
earthengine authenticate

# 3. Run the Streamlit app
cd app
streamlit run forest_disturbance_app.py
```

### Notebook
Run `notebook/himalayan_forest_disturbance_v4.ipynb` in Jupyter
after placing `gadm41_IND_shp.zip` in the same folder.
