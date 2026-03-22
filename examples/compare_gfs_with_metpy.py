from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np
import xarray as xr
from metpy.units import units


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
DEFAULT_OUT_DIR = EXAMPLES / "gfs_metpy_verify"

DEFAULT_FIELDS = {
    "u500": {
        "path": EXAMPLES / "verify_gfs_ugrd500.grib2",
        "search": "UGRD:500 mb:anl:",
        "product": "pressure",
        "var": "u",
    },
    "v500": {
        "path": EXAMPLES / "verify_gfs_vgrd500.grib2",
        "search": "VGRD:500 mb:anl:",
        "product": "pressure",
        "var": "v",
    },
    "t850": {
        "path": EXAMPLES / "verify_gfs_tmp850.grib2",
        "search": "TMP:850 mb:anl:",
        "product": "pressure",
        "var": "t",
    },
    "u850": {
        "path": EXAMPLES / "verify_gfs_ugrd850.grib2",
        "search": "UGRD:850 mb:anl:",
        "product": "pressure",
        "var": "u",
    },
    "v850": {
        "path": EXAMPLES / "verify_gfs_vgrd850.grib2",
        "search": "VGRD:850 mb:anl:",
        "product": "pressure",
        "var": "v",
    },
}


def run(cmd: list[str]) -> None:
    print(">", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare pressure-level wxforge map diagnostics with MetPy.")
    parser.add_argument("--u500", type=Path, default=DEFAULT_FIELDS["u500"]["path"])
    parser.add_argument("--v500", type=Path, default=DEFAULT_FIELDS["v500"]["path"])
    parser.add_argument("--t850", type=Path, default=DEFAULT_FIELDS["t850"]["path"])
    parser.add_argument("--u850", type=Path, default=DEFAULT_FIELDS["u850"]["path"])
    parser.add_argument("--v850", type=Path, default=DEFAULT_FIELDS["v850"]["path"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--ensure-gfs-defaults",
        action="store_true",
        help="Fetch the default GFS analysis files if they are missing.",
    )
    return parser.parse_args()


def fields_from_args(args: argparse.Namespace) -> dict[str, dict[str, object]]:
    return {
        "u500": {"path": args.u500, "search": DEFAULT_FIELDS["u500"]["search"], "product": "pressure", "var": "u"},
        "v500": {"path": args.v500, "search": DEFAULT_FIELDS["v500"]["search"], "product": "pressure", "var": "v"},
        "t850": {"path": args.t850, "search": DEFAULT_FIELDS["t850"]["search"], "product": "pressure", "var": "t"},
        "u850": {"path": args.u850, "search": DEFAULT_FIELDS["u850"]["search"], "product": "pressure", "var": "u"},
        "v850": {"path": args.v850, "search": DEFAULT_FIELDS["v850"]["search"], "product": "pressure", "var": "v"},
    }


def ensure_grib_inputs(fields: dict[str, dict[str, object]]) -> None:
    for spec in fields.values():
        if spec["path"].exists():
            continue
        run(
            [
                "cargo",
                "run",
                "-p",
                "wx-cli",
                "--",
                "fetch",
                "model-subset",
                "--model",
                "gfs",
                "--product",
                spec["product"],
                "--forecast-hour",
                "0",
                "--search",
                spec["search"],
                "--output",
                str(spec["path"]),
                "--limit",
                "1",
            ]
        )


def build_rust_outputs(fields: dict[str, dict[str, object]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    latitude_path = out_dir / "latitude_input.npy"
    longitude_path = out_dir / "longitude_input.npy"
    lat2d, lon2d = coordinate_mesh(load_cfgrib_var(fields["u500"]["path"], "u"))
    np.save(latitude_path, lat2d.astype(np.float32))
    np.save(longitude_path, lon2d.astype(np.float32))
    run(
        [
            "cargo",
            "run",
            "-p",
            "wx-cli",
            "--bin",
            "verify_gfs_maps",
            "--",
            "--u500",
            str(fields["u500"]["path"]),
            "--v500",
            str(fields["v500"]["path"]),
            "--t850",
            str(fields["t850"]["path"]),
            "--u850",
            str(fields["u850"]["path"]),
            "--v850",
            str(fields["v850"]["path"]),
            "--latitude-npy",
            str(latitude_path),
            "--longitude-npy",
            str(longitude_path),
            "--output-dir",
            str(out_dir),
        ]
    )


def load_cfgrib_var(path: Path, var: str) -> xr.DataArray:
    ds = xr.open_dataset(path, engine="cfgrib")
    return ds[var]


def coordinate_mesh(data_array: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    lat = data_array.coords["latitude"].values
    lon = data_array.coords["longitude"].values
    if lat.ndim == 2 and lon.ndim == 2:
        return lat.astype(np.float64), lon.astype(np.float64)
    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
        return lat2d.astype(np.float64), lon2d.astype(np.float64)
    raise ValueError(
        f"unsupported coordinate shapes: latitude {lat.shape}, longitude {lon.shape}"
    )


def summary_stats(reference: np.ndarray, candidate: np.ndarray, trim_edges: bool = False) -> dict[str, float]:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if trim_edges and ref.ndim == 2 and ref.shape[0] > 4 and ref.shape[1] > 4:
        ref = ref[2:-2, 2:-2]
        cand = cand[2:-2, 2:-2]
    mask = np.isfinite(ref) & np.isfinite(cand)
    if not np.any(mask):
        return {
            "count": 0,
            "mean_abs": float("nan"),
            "rmse": float("nan"),
            "max_abs": float("nan"),
            "p99_abs": float("nan"),
            "mean_bias": float("nan"),
        }
    diff = cand[mask] - ref[mask]
    abs_diff = np.abs(diff)
    return {
        "count": int(mask.sum()),
        "mean_abs": float(abs_diff.mean()),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs": float(abs_diff.max()),
        "p99_abs": float(np.percentile(abs_diff, 99)),
        "mean_bias": float(diff.mean()),
    }


def save_triptych(
    name: str, lon: np.ndarray, lat: np.ndarray, rust: np.ndarray, metpy: np.ndarray, out_dir: Path
) -> None:
    diff = rust - metpy
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]

    vmin = np.nanpercentile(np.concatenate([rust.ravel(), metpy.ravel()]), 2)
    vmax = np.nanpercentile(np.concatenate([rust.ravel(), metpy.ravel()]), 98)
    dlim = np.nanpercentile(np.abs(diff), 99)
    if not np.isfinite(dlim) or dlim == 0.0:
        dlim = 1.0

    panels = [
        ("wxforge", rust, "viridis", vmin, vmax),
        ("MetPy", metpy, "viridis", vmin, vmax),
        ("wxforge - MetPy", diff, "coolwarm", -dlim, dlim),
    ]
    for ax, (title, data, cmap, lo, hi) in zip(axes, panels, strict=True):
        im = ax.imshow(data, origin="upper", extent=extent, cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, shrink=0.9)

    fig.suptitle(name)
    fig.savefig(out_dir / f"{name}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    fields = fields_from_args(args)

    if args.ensure_gfs_defaults:
        ensure_grib_inputs(fields)
    else:
        missing = [str(spec["path"]) for spec in fields.values() if not spec["path"].exists()]
        if missing:
            raise FileNotFoundError(
                "missing input file(s); provide them explicitly or rerun with --ensure-gfs-defaults:\n"
                + "\n".join(missing)
            )

    u500_da = load_cfgrib_var(fields["u500"]["path"], "u")
    v500_da = load_cfgrib_var(fields["v500"]["path"], "v")
    t850_da = load_cfgrib_var(fields["t850"]["path"], "t")
    u850_da = load_cfgrib_var(fields["u850"]["path"], "u")
    v850_da = load_cfgrib_var(fields["v850"]["path"], "v")
    lat2d, lon2d = coordinate_mesh(u500_da)

    build_rust_outputs(fields, args.output_dir)

    u500 = u500_da.values * units("m/s")
    v500 = v500_da.values * units("m/s")
    t850 = t850_da.values * units.kelvin
    u850 = u850_da.values * units("m/s")
    v850 = v850_da.values * units("m/s")
    dx, dy = mpcalc.lat_lon_grid_deltas(lon2d, lat2d)

    metpy_products = {
        "u500": u500.magnitude,
        "v500": v500.magnitude,
        "t850": t850.magnitude,
        "u850": u850.magnitude,
        "v850": v850.magnitude,
        "div500": mpcalc.divergence(u500, v500, dx=dx, dy=dy).magnitude,
        "vort500": mpcalc.vorticity(u500, v500, dx=dx, dy=dy).magnitude,
        "tadv850": mpcalc.advection(t850, u=u850, v=v850, dx=dx, dy=dy).magnitude,
        "theta850": mpcalc.potential_temperature(850.0 * units.hPa, t850).magnitude,
    }

    rust_products = {
        name: np.load(args.output_dir / f"{name}.npy").astype(np.float64)
        for name in [
            "longitude",
            "latitude",
            "u500",
            "v500",
            "t850",
            "u850",
            "v850",
            "div500",
            "vort500",
            "tadv850",
            "theta850",
        ]
    }

    report: dict[str, dict[str, float] | dict[str, dict[str, float]]] = {}
    for name, metpy_values in metpy_products.items():
        rust_values = rust_products[name]
        report[name] = {
            "all_points": summary_stats(metpy_values, rust_values, trim_edges=False),
            "core_points": summary_stats(metpy_values, rust_values, trim_edges=True),
        }
        save_triptych(name, lon2d, lat2d, rust_values, metpy_values, args.output_dir)

    coord_report = {
        "longitude": summary_stats(lon2d, rust_products["longitude"]),
        "latitude": summary_stats(lat2d, rust_products["latitude"]),
    }

    output = {
        "coordinate_alignment": coord_report,
        "products": report,
    }
    (args.output_dir / "comparison_summary.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
