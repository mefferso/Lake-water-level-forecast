# Radar QPF Map Setup

This repo now has a standalone Leaflet viewer in `index.html`.

## GitHub Pages

To publish it:

1. Open the repo on GitHub.
2. Go to **Settings**.
3. Go to **Pages**.
4. Use:
   - Source: `Deploy from a branch`
   - Branch: `main`
   - Folder: `/root`
5. Save.

Expected URL:

`https://mefferso.github.io/Radar-QPF-map/`

## What is built in

- Leaflet map centered over southeast Louisiana / south Mississippi / south Alabama
- KHDC radar toggle
- KMOB radar toggle
- MRMS toggle
- Accumulation-period dropdown
- Opacity slider
- Base map selector
- Radar site markers
- Approximate radar range rings
- Approximate precip legend

## Important WMS note

The viewer is wired for NOAA-style WMS layers, but the exact single-site storm-total precip service/layer names may need adjustment.

The config block to edit is in `index.html`:

```js
const precipLayerConfig = {
  khdc: {
    label: "KHDC Storm Total Precip",
    url: "https://opengeo.ncep.noaa.gov/geoserver/conus/conus_bref_qcd/ows",
    layers: "conus_bref_qcd",
    radarId: "KHDC"
  },
  kmob: {
    label: "KMOB Storm Total Precip",
    url: "https://opengeo.ncep.noaa.gov/geoserver/conus/conus_bref_qcd/ows",
    layers: "conus_bref_qcd",
    radarId: "KMOB"
  },
  mrms: {
    label: "MRMS QPE / Precip Accumulation",
    url: "https://opengeo.ncep.noaa.gov/geoserver/mrms/ows",
    layers: "mrms:MultiSensor_QPE_01H_Pass2"
  }
};
```

MRMS example layer names in the app:

- `mrms:MultiSensor_QPE_01H_Pass2`
- `mrms:MultiSensor_QPE_03H_Pass2`
- `mrms:MultiSensor_QPE_06H_Pass2`
- `mrms:MultiSensor_QPE_24H_Pass2`

## Good next upgrades

- Confirm operational WMS endpoint and layer names for KHDC/KMOB storm-total precipitation
- Add timestamp / valid time display
- Add click-to-query precip value under cursor
- Add CWA/county overlays
- Add looping time-enabled QPE frames
