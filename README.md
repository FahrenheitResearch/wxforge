# wxforge

All-Rust weather and ML data pipeline. Fetch, decode, calculate, render, and export training-ready datasets from operational NWP models -- no C or Python dependencies in the core.

**22,488 lines** across **9 crates**, one CLI binary, one mission: turn raw GRIB into training data as fast as the hardware allows.

```
wxforge fetch  -->  wxforge decode  -->  wxforge calc  -->  wxforge train plan  -->  wxforge train build
                                                                                         |
                                                                            Training-ready NPY / Parquet / WebDataset
```

---

## Quick Start

### Prerequisites

- [Rust](https://rustup.rs/) (1.75+ recommended)
- Git

### Build

```bash
git clone https://github.com/your-org/wxforge.git
cd wxforge
cargo build --release
```

The release binary lands at `target/release/wxforge`.

### Run your first command

```bash
# List supported models
wxforge models

# Fetch HRRR 2-m temperature via byte-range subset (~325 KB)
wxforge fetch model-subset \
  --model hrrr --product surface --forecast-hour 0 \
  --search "TMP:2 m above ground" \
  --output hrrr_tmp2m.grib2 --limit 1

# Inspect what you downloaded
wxforge scan-grib --file hrrr_tmp2m.grib2

# Decode message 1
wxforge decode-grib --file hrrr_tmp2m.grib2 --message 1

# Render to PNG
wxforge render grib --file hrrr_tmp2m.grib2 --message 1 \
  --output hrrr_tmp2m.png --colormap heat
```

---

## CLI Reference

All commands are subcommands of the `wxforge` binary (or `cargo run -p wx-cli --bin wxforge --`).

### `wxforge models`

List every supported model with grid specs, sources, and available products.

```bash
wxforge models
```

### `wxforge fetch model-subset`

Download specific GRIB fields via HTTP byte-range requests against `.idx` inventory files. Only the bytes you need cross the wire.

```bash
# Single field
wxforge fetch model-subset \
  --model hrrr --product surface --forecast-hour 0 \
  --search "TMP:2 m above ground" \
  --output hrrr_tmp2m.grib2 --limit 1

# CAPE field (~325 KB instead of ~80 MB)
wxforge fetch model-subset \
  --model hrrr --product surface --forecast-hour 3 \
  --search "CAPE:surface" \
  --output hrrr_cape.grib2 --limit 1
```

### `wxforge fetch model-download`

Download a complete GRIB file from an operational source. Supports authenticated CDS retrievals for ERA5.

```bash
# GFS surface analysis
wxforge fetch model-download \
  --model gfs --product surface --forecast-hour 0 \
  --output gfs_surface.grib2

# ERA5 2-m temperature over CONUS
wxforge fetch model-download \
  --model era5 --product surface --forecast-hour 0 \
  --run 2024010100 --variables 2m_temperature \
  --area 55,-130,20,-60 \
  --output era5_2t.grib

# ERA5 850 hPa temperature
wxforge fetch model-download \
  --model era5 --product pressure --forecast-hour 0 \
  --run 2024010100 --variables temperature \
  --pressure-level 850 --area 55,-130,20,-60 \
  --output era5_t850.grib

# Check what's available right now
wxforge fetch model-download \
  --model gfs --product surface --forecast-hour 0 \
  --output gfs.grib2 --available
```

### `wxforge scan-grib`

List every message in a GRIB file with variable, level, and packing metadata.

```bash
wxforge scan-grib --file hrrr_tmp2m.grib2
```

### `wxforge decode-grib`

Decode a single message and print grid geometry, statistics, and sample values. Supports GRIB1 and GRIB2 (simple packing, complex+spatial differencing, JPEG 2000, CCSDS/AEC).

```bash
wxforge decode-grib --file hrrr_tmp2m.grib2 --message 1
```

### `wxforge calc thermo`

Compute thermodynamic quantities from a single observation.

```bash
wxforge calc thermo --temperature-c 20 --dewpoint-c 10 --pressure-hpa 850
```

Output includes saturation vapor pressure, mixing ratio, theta-e, LCL, wet-bulb temperature, and more. All calculations are verified against MetPy to within instrument precision.

### `wxforge render grib`

Render a decoded GRIB message to a georeferenced PNG.

```bash
wxforge render grib \
  --file era5_2t.grib --message 1 \
  --output era5_2t.png --colormap heat
```

### `wxforge train job-init`

Create an ML job specification for a given architecture and task.

```bash
wxforge train job-init \
  --output job_spec.json \
  --architecture swin-transformer \
  --task forecasting \
  --dataset-name hrrr_swin_demo
```

Supported architectures: `classical-ml`, `swin-transformer`, `diffusion`, `forecast-graph-network`, `custom`.

### `wxforge train job-plan`

Expand a job spec into a full training plan with dataset request, feature profiles, shard layout, and model recipe.

```bash
wxforge train job-plan --spec job_spec.json
```

### `wxforge train build-grib-sample`

Build NPY arrays from decoded GRIB fields, ready for ingestion by PyTorch or NumPy.

```bash
wxforge train build-grib-sample \
  --file hrrr_tmp2m.grib2 \
  --output-dir sample_bundle \
  --colormap heat
```

---

## Supported Models

| Model | Resolution | Grid | Source | Products |
|-------|-----------|------|--------|----------|
| **HRRR** | 3 km | 1799 x 1059 Lambert | NOAA NOMADS / AWS | surface, pressure |
| **GFS** | 0.25 deg | 1440 x 721 lat-lon | NOAA NOMADS | surface, pressure |
| **NAM** | 12 km | 614 x 428 Lambert | NOAA NOMADS | surface, pressure |
| **RAP** | 13 km | 337 x 451 Lambert | NOAA NOMADS | surface, pressure |
| **ECMWF IFS** | 0.25 deg | 1440 x 721 lat-lon | ECMWF Open Data | surface, pressure |
| **ERA5** | 0.25 deg | 1440 x 721 lat-lon | CDS API | single-level, pressure-level |

HRRR, GFS, NAM, and RAP use NOAA `.idx` inventory files for byte-range subsetting. ECMWF IFS uses `.index` inventories. ERA5 uses authenticated CDS API retrieval.

CDS credentials are discovered from environment variables (`CDSAPI_URL` / `CDSAPI_KEY`) or config files (`~/.cdsapirc`, `~/.ecmwfdatastoresrc`).

---

## Architecture

wxforge is a Cargo workspace of 9 crates, each with a single responsibility:

| Crate | Lines | Role |
|-------|-------|------|
| **wx-types** | ~150 | Shared domain model: grids, fields, soundings, radar volumes, dataset specs |
| **wx-grib** | ~2,800 | GRIB1/GRIB2 scanning, inventory parsing, message decode (simple, complex+spatial, JPEG 2000, CCSDS) |
| **wx-fetch** | ~1,300 | Download planning, source templates, byte-range fetch, cache semantics, CDS retrieval |
| **wx-calc** | ~7,400 | Thermodynamics, kinematics, severe weather indices -- verified against MetPy |
| **wx-radar** | ~160 | NEXRAD palette parsing, value sampling, color table decode |
| **wx-render** | ~130 | Deterministic PNG raster rendering with configurable colormaps |
| **wx-export** | ~280 | Dataset manifests, export targets (Arrow, Parquet, WebDataset, Zarr) |
| **wx-train** | ~950 | ML job specs, training plan expansion, NPY/shard dataset assembly |
| **wx-cli** | ~4,800 | Single binary entrypoint -- all commands, argument parsing, orchestration |

### Design principles

- **Zero C/Python dependencies** in the core pipeline
- **Verified calculations** -- thermo functions regression-tested against MetPy
- **Deterministic rendering** -- same input always produces the same PNG
- **Training export is first-class** -- not bolted on after the fact

---

## Training Pipeline

The training pipeline supports end-to-end workflows from raw GRIB to GPU-ready tensors.

### Architecture-aware job planning

Each ML architecture gets tailored defaults for feature profiles, export format, shard layout, and parallelism:

| Architecture | Export Format | Features | Shard Size |
|-------------|-------------|----------|------------|
| Classical ML (XGBoost) | Parquet | surface_core, severe_diagnostics, tabular_stats | 16 samples |
| Swin Transformer | WebDataset | surface_core, pressure_core, severe_diagnostics | 96 samples |
| Diffusion (UNet/DiT) | WebDataset | surface_core, pressure_core, thermodynamic_profiles | 128 samples |
| Forecast Graph Network | Parquet | surface_core, pressure_core | 48 samples |

### Example workflow

```bash
# 1. Create a job spec
wxforge train job-init \
  --output job.json \
  --architecture swin-transformer \
  --task forecasting \
  --dataset-name hrrr_forecast

# 2. Expand to full training plan
wxforge train job-plan --spec job.json

# 3. Build dataset from GRIB files
wxforge train build-grib-sample \
  --file hrrr_surface.grib2 \
  --output-dir training_data/ \
  --colormap heat
```

### Job spec files

Job specs are JSON files that declaratively define a training run. See `examples/agent_job_swin.json` and `examples/agent_job_classical.json` for complete examples.

---

## Python Integration

The `wxforge-data` package provides PyTorch dataset loaders for wxforge-exported training data. It reads NPY/JSON artifacts produced by `wxforge train build-*` and presents them as standard `torch.utils.data.Dataset` objects.

```bash
pip install -e python/
```

```python
from wxforge_data import WxforgeDataset, WxforgeMultiSampleDataset

# Single sample bundle
ds = WxforgeDataset("training_data/sample_bundle")

# Multi-sample dataset
ds = WxforgeMultiSampleDataset("training_data/")
```

The Python package does **not** depend on the wxforge binary at runtime.

---

## Benchmarks

Measured on a desktop workstation (Windows 11, NVMe SSD, residential broadband):

| Operation | Result |
|-----------|--------|
| Fetch HRRR CAPE via `.idx` byte-range | ~325 KB transferred in ~1 s |
| Decode 1799 x 1059 GRIB2 message | Instant (< 10 ms) |
| Full E2E: fetch, decode, plan, build | < 5 s |
| 24 forecast hours x 9 fields | ~80 MB total transfer |

Byte-range subsetting is the key: a full HRRR surface file is ~80 MB, but a single field is 300-400 KB. The `.idx` inventory tells wxforge exactly which bytes to request.

---

## Verification

wxforge includes a comprehensive verification suite that regression-tests thermodynamic calculations against MetPy:

```bash
# Run the MetPy regression suite
python examples/metpy_regression_suite.py

# Run the full verification suite
python examples/verification_suite.py

# Benchmark the pipeline
python examples/benchmark_suite.py
```

Reports are written to `examples/` as JSON for automated comparison.

---

## Roadmap

- Broader GRIB2 template coverage (PNG packing, run-length)
- NEXRAD Level-II volume ingest and derived products
- Expanded calc parity (vorticity, frontogenesis, PV)
- Zarr and WebDataset export with streaming writes
- Distributed fetch orchestration for multi-cycle bulk downloads
- Python bindings via PyO3 for hybrid workflows
- GPU-accelerated decode and rendering

---

## Documentation

Detailed docs live in the `docs/` directory:

- [Architecture](docs/ARCHITECTURE.md) -- crate graph and data flow
- [Agent Jobs](docs/AGENT_JOBS.md) -- ML job spec format and planning
- [Calc Parity](docs/CALC_PARITY.md) -- MetPy verification status
- [MetPy Regression](docs/METPY_REGRESSION.md) -- regression test results
- [Model Coverage](docs/MODEL_COVERAGE.md) -- per-model fetch/decode/render status
- [Reference Matrix](docs/REFERENCE_MATRIX.md) -- mapping to predecessor repos

---

## Credits

Built with [Codex](https://openai.com/codex). Thermodynamic calculations verified against [MetPy](https://unidata.github.io/MetPy/).

## License

MIT
