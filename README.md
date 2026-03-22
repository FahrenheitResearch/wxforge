# wxforge

`wxforge` is a brand-new Rust monorepo for an end-to-end weather and ML data pipeline.

This repo is intentionally separate from the existing reference repos:

- [metrust-py](C:/Users/drew/metrust-py)
- [rustmet](C:/Users/drew/rustmet)
- [rustdar](C:/Users/drew/rustdar)
- [rusbie](C:/Users/drew/rusbie)
- [cfrust](C:/Users/drew/cfrust)
- [rustplots](C:/Users/drew/rustplots)
- [rustmet-train](C:/Users/drew/rustmet-train)

Those projects are treated as references, regression targets, and idea mines. They are not modified by this repo.

## Mission

Build one canonical Rust crate graph that can:

1. fetch weather data from operational and archival sources
2. decode GRIB and radar formats without C or Python dependencies
3. run scientifically verified meteorological calculations
4. render deterministic products for diagnostics and ML
5. export training-ready datasets in Rust-native pipelines
6. expose a single CLI and agent-friendly orchestration surface

## Crate Layout

| Crate | Role |
|---|---|
| `wx-types` | shared domain model for grids, fields, soundings, radar, and dataset specs |
| `wx-grib` | GRIB inventory, message selection, and decode interfaces |
| `wx-fetch` | download planning, source templates, cache semantics, byte-range fetch contracts |
| `wx-calc` | canonical meteorological calculation layer and parity policy |
| `wx-radar` | NEXRAD and radar-specific ingest, derived products, color tables, detection APIs |
| `wx-render` | deterministic raster/vector/chart rendering for tiles, PNG, and ML surfaces |
| `wx-export` | dataset manifests and export targets such as Arrow, Parquet, WebDataset, and Zarr |
| `wx-train` | weather-to-training-data planning and dataset assembly |
| `wx-cli` | the single executable and operator-facing entrypoint |

## Principles

- One source of truth for every domain concern
- Rust-first internals, Python only as an adapter later if needed
- Verified calc quality is non-negotiable
- Rendering is deterministic and testable
- Radar is a first-class data type, not an afterthought
- Training export is part of the platform, not a side project

## Current Status

`wxforge` is beyond the pure scaffold stage. The repo now includes fresh Rust code for:

- model and source fetch planning with latest-cycle logic
- direct model download execution, including authenticated CDS retrievals
- local and HTTP byte-range fetch with subset stitching from `.idx` inventories
- wgrib2-style `.idx` inventory parsing and search
- native GRIB1 and GRIB2 message scanning and metadata extraction
- native GRIB1 simple-packed regular lat/lon decode
- native GRIB2 field decode for simple-packed, IEEE-float, operational template-3 complex+spatial, template-40 JPEG 2000, and template-42 CCSDS/AEC messages
- core thermo and regular-grid kinematics demos
- radar palette parsing and value sampling
- deterministic PNG raster rendering
- dataset manifest generation plus `.npy` training-sample bundles from decoded GRIB fields
- a single CLI that exercises all of the above

This is still the start of the rewrite, not the final system. The next heavy lifts are broader GRIB coverage, radar volume ingest, verified calc parity expansion, and richer Rust-native dataset builders.

Major-model status is tracked in [Model Coverage](C:/Users/drew/wxforge/docs/MODEL_COVERAGE.md). Right now:

- `HRRR`, `RAP`, `NAM`, and `GFS` are live-tested through planning, subset fetch, decode, render, and training-bundle export
- `ECMWF IFS` open data is live-tested through planning, `.index` inventory parsing, subset fetch, decode, render, and training-bundle export
- `ERA5` single-level and pressure-level data are live-tested through authenticated CDS retrieval, GRIB1 decode, render, and training-bundle export

CDS credentials are discovered from:

- `ECMWF_DATASTORES_URL` / `ECMWF_DATASTORES_KEY`
- `CDSAPI_URL` / `CDSAPI_KEY`
- local `~/.ecmwfdatastoresrc` or `~/.cdsapirc`
- WSL `~/.ecmwfdatastoresrc` or `~/.cdsapirc` when running on Windows

## Commands

```powershell
cargo run -p wx-cli -- about
cargo run -p wx-cli -- crates
cargo run -p wx-cli -- models
cargo run -p wx-cli -- fetch subset --grib examples\sample.grib2 --idx examples\sample.idx --search "2 m above ground" --output examples\subset_2m.grib2
cargo run -p wx-cli -- fetch model-download --model gfs --product surface --forecast-hour 0 --output examples\gfs_surface.grib2 --available
cargo run -p wx-cli -- fetch model-subset --model hrrr --product surface --forecast-hour 0 --search "TMP:2 m above ground" --output examples\hrrr_auto_subset.grib2 --limit 1
cargo run -p wx-cli -- fetch model-download --model era5 --product surface --forecast-hour 0 --run 2024010100 --variables 2m_temperature --area 55,-130,20,-60 --output examples\era5_2t_subset.grib
cargo run -p wx-cli -- fetch model-download --model era5 --product pressure --forecast-hour 0 --run 2024010100 --variables temperature --pressure-level 850 --area 55,-130,20,-60 --output examples\era5_t850_subset.grib
cargo run -p wx-cli -- plan-fetch --model hrrr --product surface --forecast-hour 3
cargo run -p wx-cli -- plan-fetch --model hrrr --product surface --forecast-hour 0 --available
cargo run -p wx-cli -- parse-idx --file examples\sample.idx
cargo run -p wx-cli -- scan-grib --file examples\sample.grib2
cargo run -p wx-cli -- decode-grib --file examples\sample.grib2 --message 1
cargo run -p wx-cli -- decode-grib --file examples\era5_2t_subset.grib --message 1
cargo run -p wx-cli -- calc parity --limit 20
cargo run -p wx-cli -- calc thermo --temperature-c 20 --dewpoint-c 10 --pressure-hpa 850
cargo run -p wx-cli -- calc grid-demo --nx 5 --ny 5 --dx-m 1000 --dy-m 1000
cargo run -p wx-cli -- radar parse-palette --file examples\sample.pal --value 55
cargo run -p wx-cli -- render gradient --output examples\gradient.png --colormap radar
cargo run -p wx-cli -- render grib --file examples\era5_2t_subset.grib --message 1 --output examples\era5_2t_subset.png --colormap heat
cargo run -p wx-cli -- train plan --preset radar
cargo run -p wx-cli --bin wxforge -- train job-init --output examples\agent_job_swin.json --architecture swin-transformer --task forecasting --dataset-name hrrr_swin_demo
cargo run -p wx-cli --bin wxforge -- train job-plan --spec examples\agent_job_swin.json
cargo run -p wx-cli --bin wxforge -- train job-build --spec examples\agent_job_classical.json --output-dir examples\agent_classical_build --colormap heat
cargo run -p wx-cli -- train build-grib-sample --file examples\sample.grib2 --output-dir examples\sample_bundle --colormap heat
cargo run -p wx-cli -- train build-grib-sample --file examples\era5_2t_subset.grib --output-dir examples\era5_2t_bundle --messages 1 --colormap heat
cargo run -p wx-cli -- train build-grib-dataset --manifest examples\sample_dataset_manifest.json --output-dir examples\sample_batch_dataset --colormap heat
python examples\metpy_regression_suite.py
python examples\compare_thermo_profiles_with_metpy.py
python examples\benchmark_suite.py
python examples\verify_ml_pipeline.py
python examples\verification_suite.py
cargo test --workspace
powershell -NoProfile -File examples\make_sample_grib2.ps1
powershell -NoProfile -File examples\make_agent_job_sample.ps1
```

## Docs

- [Architecture](C:/Users/drew/wxforge/docs/ARCHITECTURE.md)
- [Agent Jobs](C:/Users/drew/wxforge/docs/AGENT_JOBS.md)
- [Calc Parity](C:/Users/drew/wxforge/docs/CALC_PARITY.md)
- [MetPy Regression](C:/Users/drew/wxforge/docs/METPY_REGRESSION.md)
- [Reference Matrix](C:/Users/drew/wxforge/docs/REFERENCE_MATRIX.md)
- [Migration Plan](C:/Users/drew/wxforge/docs/MIGRATION_PLAN.md)
- [Model Coverage](C:/Users/drew/wxforge/docs/MODEL_COVERAGE.md)
