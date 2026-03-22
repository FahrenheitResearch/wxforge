# Model Coverage

This file tracks what `wxforge` can currently do for major training-relevant models.

Status labels:

- `planned`: URL/source planning is implemented
- `inventory`: remote inventory parsing and message search work
- `subset`: byte-range subset fetch works
- `decode`: GRIB decode works for at least one live field
- `blocked`: a known hard blocker remains

## Live-Checked Coverage

These checks were run against live sources on `2026-03-16`.

| Model | Source | Planning | Inventory | Subset | Decode | Notes |
|---|---|---|---|---|---|---|
| HRRR | NOMADS | yes | `.idx` | yes | yes | Live `2t` subset decoded cleanly |
| RAP | NOMADS | yes | `.idx` | yes | yes | Live `2t` subset decoded cleanly via template `40` JPEG 2000 |
| NAM | NOMADS | yes | `.idx` | yes | yes | Live `2t` subset decoded cleanly |
| GFS | NOMADS | yes | `.idx` | yes | yes | Live `2t` subset decoded cleanly |
| ECMWF IFS | ECMWF Open Data | yes | `.index` JSON-lines | yes | yes | Live `2t` subset decoded cleanly via template `42` CCSDS/AEC |
| ERA5 single levels | CDS API | yes | n/a | n/a | yes | Live authenticated `2m_temperature` retrieval, GRIB1 decode, render, and bundle export work |
| ERA5 pressure levels | CDS API | yes | n/a | n/a | yes | Live authenticated `temperature@850 hPa` retrieval, GRIB1 decode, render, and bundle export work |

## Current Interpretation

- NOAA operational training models are in good shape for `HRRR`, `RAP`, `NAM`, and `GFS`.
- `ECMWF IFS` open data is now fetchable and decodable through the Rust pipeline.
- `ERA5` is now a real authenticated CDS flow in Rust for the single-level and pressure-level datasets.
- `ERA5` no longer uses public byte-range subset logic; it uses request-driven CDS retrieval followed by local decode/export.

## Concrete Blockers

### RAP

- Live subset scans correctly as a GRIB2 Lambert Conformal field.
- Decode now works through template `40` using a pure-Rust JPEG 2000 decoder.
- Training bundle export also works from the decoded subset.

### ECMWF Open Data

- Live planning works against `data.ecmwf.int`.
- Live `.index` JSON-lines parsing works.
- Live subset fetch works.
- Decode now works through template `42` using a pure-Rust CCSDS/AEC decoder.
- Training bundle export also works from the decoded subset.

### ERA5

- `wx-fetch` now plans `ERA5` retrievals as authenticated `cds://...` requests and executes them through the CDS Retrieve API.
- Credentials are discovered from env vars, local config files, or WSL config files on Windows.
- Live retrieval is verified for:
  - `reanalysis-era5-single-levels`
  - `reanalysis-era5-pressure-levels`
- The retrieved outputs are currently `GRIB1`, and `wx-grib` now decodes that operational simple-packed regular lat/lon path.
- The `reanalysis-era5-complete` dataset is still only planner-level and has not been live-verified here.

## Practical Next Steps

1. Expand regression coverage beyond `2t` to include additional RAP, ECMWF, and ERA5 parameters.
2. Add more GRIB1/GRIB2 templates and grid-template regressions now that both CDS and open-data sources are live.
3. Extend ERA5 live coverage to `reanalysis-era5-complete` if that dataset becomes training-critical.
