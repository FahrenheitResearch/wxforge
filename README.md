# wxtrain

All-Rust weather data pipeline for ML training. Fetches GRIB from operational NWP models, decodes natively, computes derived fields, and exports training-ready datasets.

27,559 lines of Rust across 9 crates. Single binary. One `cargo build --release`. No eccodes, no Fortran, no C dependencies.

## Quick Start

```bash
git clone <repo-url>
cd wxtrain
cargo build --release
```

The binary lands at `target/release/wxtrain`.

```bash
# List supported models
wxtrain models

# Fetch a single HRRR field via byte-range subset (~325 KB instead of ~80 MB)
wxtrain fetch model-subset \
  --model hrrr --product surface --forecast-hour 0 \
  --search "TMP:2 m above ground" \
  --output hrrr_tmp2m.grib2 --limit 1

# See what's in it
wxtrain scan-grib --file hrrr_tmp2m.grib2

# Decode message 1
wxtrain decode-grib --file hrrr_tmp2m.grib2 --message 1

# Render to PNG
wxtrain render grib --file hrrr_tmp2m.grib2 --message 1 \
  --output hrrr_tmp2m.png --colormap heat

# Download full GRIBs in parallel (24 forecast hours, 4 concurrent)
wxtrain fetch batch \
  --model hrrr --product surface \
  --forecast-hours 0-23 --output-dir ./hrrr_data/ --parallelism 4

# Compute thermodynamic quantities
wxtrain calc thermo --temperature-c 20 --dewpoint-c 10 --pressure-hpa 850
```

## Supported Models

| Model | Resolution | Grid | Source | Subsetting |
|-------|-----------|------|--------|------------|
| HRRR | 3 km | 1799x1059 Lambert | NOAA NOMADS / AWS | `.idx` byte-range |
| GFS | 0.25 deg | 1440x721 lat-lon | NOAA NOMADS | `.idx` byte-range |
| NAM | 12 km | 614x428 Lambert | NOAA NOMADS | `.idx` byte-range |
| RAP | 13 km | 337x451 Lambert | NOAA NOMADS | `.idx` byte-range |
| ECMWF IFS | 0.25 deg | 1440x721 lat-lon | ECMWF Open Data | `.index` inventory |
| ERA5 | 0.25 deg | 1440x721 lat-lon | CDS API | Authenticated retrieval |

Byte-range subsetting uses `.idx` inventory files to request only the fields you need. A single HRRR surface field is 300-400 KB; the full file is ~80 MB.

ERA5 credentials are discovered from `CDSAPI_URL`/`CDSAPI_KEY` environment variables or `~/.cdsapirc`.

## Architecture

| Crate | Lines | Role |
|-------|-------|------|
| `wx-types` | 153 | Domain model: grids, fields, soundings, radar volumes, dataset specs |
| `wx-grib` | 2,814 | GRIB1/GRIB2 scanner, inventory parser, message decoder (simple, complex+spatial differencing, JPEG 2000, CCSDS/AEC) |
| `wx-fetch` | 1,305 | Download planning, source templates, byte-range fetch, cache semantics, CDS retrieval |
| `wx-calc` | 7,391 | Thermodynamics, kinematics, severe weather indices -- 199 functions verified against MetPy |
| `wx-radar` | 159 | NEXRAD palette parsing, value sampling, color table decode |
| `wx-render` | 133 | Deterministic PNG rasterization with configurable colormaps |
| `wx-export` | 283 | Dataset manifests, export targets (Arrow, Parquet, WebDataset, Zarr) |
| `wx-train` | 948 | ML job specs, training plan expansion, NPY/shard dataset assembly |
| `wx-cli` | 5,057 | Single binary entrypoint -- all commands, argument parsing, orchestration |

## Training Pipeline

The pipeline goes from raw GRIB to GPU-ready tensors. Job planning is architecture-aware:

| Architecture | Export Format | Shard Size |
|-------------|-------------|------------|
| Classical ML (XGBoost) | Parquet | 16 samples |
| Swin Transformer | WebDataset | 96 samples |
| Diffusion (UNet/DiT) | WebDataset | 128 samples |
| Forecast Graph Network | Parquet | 48 samples |

### Workflow

```bash
# 1. Create a job spec
wxtrain train job-init \
  --output job.json \
  --architecture swin-transformer \
  --task forecasting \
  --dataset-name hrrr_forecast

# 2. Expand to full training plan (features, shards, model recipe)
wxtrain train job-plan --spec job.json

# 3. Build NPY arrays from GRIB files
wxtrain train build-grib-sample \
  --file hrrr_surface.grib2 \
  --output-dir training_data/ \
  --colormap heat
```

Supported architectures: `classical-ml`, `swin-transformer`, `diffusion`, `forecast-graph-network`, `custom`.

### Working Training Examples

Four end-to-end examples in `examples/training/`, each tested on a fresh Linux node (clone, build, fetch, train, inference) on a Blackwell GPU:

| Script | Task | Architecture |
|--------|------|-------------|
| `train_unet.py` | CAPE regression from surface fields | UNet (3-level encoder/decoder + skip connections) |
| `train_classifier.py` | Severe weather classification | MLP classifier |
| `train_swin.py` | 3-hour forecast | Swin Transformer |
| `train_diffusion.py` | Super-resolution | Diffusion model |

Each script calls `wxtrain fetch batch` and `wxtrain train build-grib-sample` to build its own dataset, then trains with PyTorch.

## Python Integration

The `wxtrain_data` package provides PyTorch dataset loaders for wxtrain-exported data.

```bash
pip install -e python/
```

```python
from wxtrain_data import WxforgeDataset, WxforgeMultiSampleDataset

# Single sample bundle
ds = WxforgeDataset("training_data/sample_bundle")

# Multi-sample dataset
ds = WxforgeMultiSampleDataset("training_data/")
```

The Python package reads NPY/JSON artifacts produced by `wxtrain train build-*` and does not depend on the wxtrain binary at runtime.

## CLI Reference

| Command | Description |
|---------|-------------|
| `wxtrain models` | List all supported models with grid specs and sources |
| `wxtrain fetch model-subset` | Download specific fields via byte-range `.idx` requests |
| `wxtrain fetch model-download` | Download a complete GRIB file (supports ERA5 CDS auth) |
| `wxtrain fetch batch` | Download full GRIBs for multiple forecast hours in parallel |
| `wxtrain scan-grib` | List every message in a GRIB file |
| `wxtrain decode-grib` | Decode a message and print grid geometry, stats, sample values |
| `wxtrain calc thermo` | Compute thermodynamic quantities from a single observation |
| `wxtrain render grib` | Render a decoded GRIB message to georeferenced PNG |
| `wxtrain train job-init` | Create an ML job spec for a given architecture and task |
| `wxtrain train job-plan` | Expand a job spec into a full training plan |
| `wxtrain train build-grib-sample` | Build NPY arrays from decoded GRIB fields |

## Timings

Measured on a desktop (Windows 11, NVMe SSD, residential internet):

| Operation | Result |
|-----------|--------|
| Fetch single HRRR field via `.idx` | ~325 KB in ~1s |
| Decode 1799x1059 GRIB2 message | <10ms |
| Full end-to-end (fetch + decode + plan + build) | <5s |
| 24 forecast hours x 9 fields | ~80 MB total transfer |

## Verification

Thermodynamic calculations are regression-tested against MetPy:

```bash
python examples/metpy_regression_suite.py
python examples/verification_suite.py
```

Reports are written to `examples/` as JSON.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) -- crate graph and data flow
- [Agent Jobs](docs/AGENT_JOBS.md) -- ML job spec format and planning
- [Calc Parity](docs/CALC_PARITY.md) -- MetPy verification status
- [MetPy Regression](docs/METPY_REGRESSION.md) -- regression test results
- [Model Coverage](docs/MODEL_COVERAGE.md) -- per-model fetch/decode/render status
- [Reference Matrix](docs/REFERENCE_MATRIX.md) -- mapping to predecessor repos

## Credits

Thermodynamic calculations verified against [MetPy](https://unidata.github.io/MetPy/). Color tables by [Solarpower07](https://github.com/Solarpower07).

## License

MIT
