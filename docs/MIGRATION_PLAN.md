# Migration Plan

## Non-Goal

This repo does not modify any reference repo.

All existing projects remain untouched and usable as:

- validation baselines
- idea sources
- migration references

## Phase 1: Stabilize The Skeleton

Done in this repo:

- workspace structure
- crate ownership model
- docs that define boundaries
- working CLI commands for fetch planning, inventory parsing, basic calc demos, rendering, radar palette parsing, and training manifests

## Phase 2: Ground Truth First

Port into `wx-calc`:

- thermodynamics
- kinematics
- severe diagnostics
- interpolation and regridding primitives

Do this only with accompanying parity tests derived from `metrust-py`.

## Phase 3: Decode Foundation

Port into `wx-grib`:

- message catalog
- inventory
- decode path
- grid template support
- packing support

Use `cfrust` fixtures as the acceptance set.

## Phase 4: Source Planning

Port into `wx-fetch`:

- source templates
- latest-run logic
- byte-range selection
- caching contracts

Use `rusbie` and `rustmet` as references, but unify on one Rust-native API.

## Phase 5: Radar

Port into `wx-radar`:

- Level II parsing
- derived radar products
- meso and TVS detection
- color tables
- sweep and volume abstractions

Use `rustdar` ideas, but keep the crate headless and reusable.

## Phase 6: Rendering

Port into `wx-render`:

- raster and contour rendering
- sounding diagrams
- radar PPI rendering
- colormap and overlay stack
- deterministic PNG output

Use `rustplots` as the primary reference and `rustdar` for radar-specific render ideas.

## Phase 7: Export And Training

Implement in `wx-export` and `wx-train`:

- dataset manifests
- channel and label specs
- Arrow, Parquet, WebDataset, and Zarr output
- benchmark dataset generation
- manifest-driven batch GRIB dataset assembly

Use `rustmet-train` only as a workflow reference.

## Phase 8: One Operator Surface

Build out `wx-cli` so an operator or agent can drive:

- ingest
- decode
- calc
- radar
- render
- export
- train-data assembly

## Definition Of Done

The rewrite is successful when:

- one monorepo owns the entire crate graph
- verified calc parity is preserved
- no duplicated foundational crates exist outside the workspace
- a single CLI can run end-to-end workflows
- Python adapters are optional and thin
