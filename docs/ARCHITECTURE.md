# Architecture

## Goal

`wxforge` is the canonical Rust workspace for end-to-end weather ingest, processing, radar analysis, rendering, and ML dataset export.

It replaces the current multi-repo drift problem with one crate graph and one ownership model.

## What This Repo Is Not

- It is not a Python drop-in compatibility project.
- It is not a UI-first workstation.
- It is not a benchmark scrapbook.
- It is not a monolith that mixes core libraries with server and notebook concerns.

Adapters can exist later, but the core must stay Rust-native and library-first.

## Layering

### 1. Domain Layer

Owned by [wx-types](C:/Users/drew/wxforge/crates/wx-types/src/lib.rs)

This crate defines the shared weather data model:

- grids and projections
- fields and units metadata
- soundings and vertical profiles
- radar volumes, sweeps, and derived products
- training manifest primitives

No network, no file decoding, no rendering.

### 2. Ingest Layer

Owned by [wx-fetch](C:/Users/drew/wxforge/crates/wx-fetch/src/lib.rs) and [wx-grib](C:/Users/drew/wxforge/crates/wx-grib/src/lib.rs)

`wx-fetch` decides where to get bytes.

`wx-grib` decides how to inventory and decode them.

This split matters:

- source logic changes frequently
- format logic must stay stable and heavily tested

### 3. Scientific Layer

Owned by [wx-calc](C:/Users/drew/wxforge/crates/wx-calc/src/lib.rs)

This is the canonical numerical truth layer.

Port here only after parity is proven against trusted references. Every calc should carry an explicit validation tier.

### 4. Radar Layer

Owned by [wx-radar](C:/Users/drew/wxforge/crates/wx-radar/src/lib.rs)

Radar is treated as a first-class subsystem with its own:

- format support
- derived products
- color table transforms
- meso/TVS/cell detection
- sweep and volume abstractions

### 5. Rendering Layer

Owned by [wx-render](C:/Users/drew/wxforge/crates/wx-render/src/lib.rs)

This crate produces deterministic outputs:

- tiles
- PNG rasters
- sounding diagrams
- radar PPIs
- overlays suitable for diffusion or segmentation pipelines

### 6. Export Layer

Owned by [wx-export](C:/Users/drew/wxforge/crates/wx-export/src/lib.rs)

This crate owns training-data output formats and dataset manifests.

The export layer should know nothing about NEXRAD internals or GRIB byte parsing. It only sees normalized domain objects.

### 7. Pipeline Layer

Owned by [wx-train](C:/Users/drew/wxforge/crates/wx-train/src/lib.rs)

This is where "turn weather data into model-ready examples" lives:

- channel definitions
- label definitions
- crop specs
- negative/positive case balancing
- benchmark dataset builders

### 8. Orchestration Layer

Owned by [wx-cli](C:/Users/drew/wxforge/crates/wx-cli/src/main.rs)

This is the one executable an operator or agent uses.

Servers, SDKs, or notebook adapters should be separate entrypoints later, but they should all consume the same library crates.

## Ownership Rules

- Only one crate owns each concept.
- No duplicated "same-name" crates across repos.
- Any future Python package must wrap these crates, not re-implement them.
- UI and server concerns must never dictate the core crate API.

## Initial Porting Strategy

1. expand verified calc formulas and parity tests in `wx-calc`
2. keep expanding `wx-grib` from today's GRIB1/GRIB2 operational coverage toward broader template and grid support
3. extend `wx-fetch` from planning plus direct download into full cache, archive, and retry orchestration
4. add radar volume parsing and detection to `wx-radar`
5. add chart and radar renderers to `wx-render`
6. add dataset writers and example builders to `wx-export` and `wx-train`

## End State

The desired end state is a rented node running one Rust-native stack where an agent can:

- fetch weather data
- compute custom diagnostics
- render and inspect outputs
- build training datasets
- iterate with minimal glue code

That stack should not depend on the historical layout of any of the reference repos.
