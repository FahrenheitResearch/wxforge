# MetPy Regression

`wxforge` now carries a thresholded regression suite against `MetPy` for the highest-value
map and severe diagnostics that were recently ported and tuned.

Current entrypoint:

```powershell
python examples\metpy_regression_suite.py
```

This suite runs the existing comparison harnesses and then enforces fixed tolerances:

- [compare_gfs_with_metpy.py](C:/Users/drew/wxforge/examples/compare_gfs_with_metpy.py)
- [compare_severe_profiles_with_metpy.py](C:/Users/drew/wxforge/examples/compare_severe_profiles_with_metpy.py)
- [compare_thermo_profiles_with_metpy.py](C:/Users/drew/wxforge/examples/compare_thermo_profiles_with_metpy.py)

Current covered cases:

- `gfs_maps`: large-area GFS analysis map parity for decoded fields and derivative products
- `gfs_maps_f006`: large-area GFS `+6h` pressure-map parity
- `gfs_maps_f012`: large-area GFS `+12h` pressure-map parity
- `era5_maps`: ERA5 pressure-level map parity on a CONUS crop
- `ecmwf_maps`: ECMWF open-data pressure-level map parity
- `severe_plains`: cool-season/plains severe-profile parity
- `severe_gulf`: warm/moist Gulf severe-profile parity
- `severe_southeast_f006`: nocturnal Southeast `GFS +6h` severe-profile parity
- `severe_florida_elevated_f012`: elevated Florida `GFS +12h` severe-profile parity
- `thermo_profiles`: fixed sounding thermo/profile parity for standalone calc APIs
- `hrrr_decode`: HRRR `2 m temperature` decode parity
- `rap_decode`: RAP `2 m temperature` decode parity
- `nam_decode`: NAM `2 m temperature` decode parity
- `ecmwf_decode`: ECMWF open-data surface-field decode parity
- `era5_decode`: ERA5 surface-field decode parity

The suite writes:

- combined report: [metpy_regression_report.json](C:/Users/drew/wxforge/examples/metpy_regression_report.json)
- per-case comparison summaries in each example output directory

Useful flags:

```powershell
python examples\metpy_regression_suite.py --case gfs_maps
python examples\metpy_regression_suite.py --case severe_plains --case severe_gulf
python examples\metpy_regression_suite.py --case thermo_profiles
python examples\metpy_regression_suite.py --no-run
```

## Purpose

This is not meant to prove all `199` public calc APIs numerically yet. It exists to lock the
current verified map, severe-diagnostic, and decoder surfaces so future refactors cannot silently regress:

- multi-template GRIB decode on operational model subsets
- spherical derivative operators
- advection / divergence / vorticity map products
- parcel thermodynamics
- CAPE / CIN
- LI / Showalter / K-Index / Total Totals / PWAT
- wet-bulb temperature / wet-bulb potential temperature / DCAPE
- Bunkers motion
- bulk shear
- storm-relative helicity
- STP

## Expansion Path

The next expansion should add more regimes and products, not loosen the current tolerances:

1. more forecast hours for the same products
2. more models (`HRRR`, `RAP`, `NAM`, `ERA5`)
3. more severe regimes (elevated convection, high terrain, cold season, tropical moisture)
4. targeted parity suites for additional `wx-calc` categories that still rely only on unit tests
