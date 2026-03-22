# Calc Parity

`wxtrain` is intended to port the full verified `metrust`/`MetPy` calc surface into native Rust.

The canonical target inventory lives in:

- [calc_inventory.txt](C:/Users/drew/wxtrain/crates/wx-calc/src/calc_inventory.txt)
- [ported_names.txt](C:/Users/drew/wxtrain/crates/wx-calc/src/ported_names.txt)
- [inventory.rs](C:/Users/drew/wxtrain/crates/wx-calc/src/inventory.rs)

The inventory is derived from the public `metrust.calc` export surface, not from ad hoc memory.

## Current State

The Rust crate now tracks `199` public calc-surface targets, including aliases and dataset/interpolation helpers.

Current command:

```powershell
cargo run -p wx-cli -- calc parity --limit 20
```

Current summary:

- `199` ported
- `0` missing

Category snapshot:

- `thermo`: `82 / 82`
- `wind`: `11 / 11`
- `kinematics`: `26 / 26`
- `severe`: `13 / 13`
- `grid`: `14 / 14`
- `atmo`: `9 / 9`
- `smooth`: `12 / 12`
- `utils`: `9 / 9`
- `dataset`: `3 / 3`
- `interpolation`: `16 / 16`

## Porting Rule

`wx-calc` should not invent a new scientific surface. It should port the verified logic from [metrust-py](C:/Users/drew/metrust-py/python/metrust/calc/__init__.py) and then carry parity tests into `wxtrain`.

## Regression Status

The parity target is no longer just public API coverage. The active requirement is numerical
agreement against `MetPy` on representative map and severe-weather cases.

Current regression entrypoint:

```powershell
python examples\metpy_regression_suite.py
```

Current regression docs:

- [MetPy Regression](C:/Users/drew/wxtrain/docs/METPY_REGRESSION.md)
