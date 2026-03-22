from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import metrust.calc as mrcalc
import numpy as np
import xarray as xr
from metpy.units import units


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
OUT_DIR = EXAMPLES / "gfs_metrust_verify"

FIELDS = {
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


def ensure_grib_inputs() -> None:
    for spec in FIELDS.values():
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


def load_cfgrib_var(path: Path, var: str) -> xr.DataArray:
    ds = xr.open_dataset(path, engine="cfgrib")
    return ds[var]


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


def save_triptych(name: str, lon: np.ndarray, lat: np.ndarray, metrust: np.ndarray, metpy: np.ndarray) -> None:
    diff = metrust - metpy
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]

    merged = np.concatenate([metrust.ravel(), metpy.ravel()])
    finite = merged[np.isfinite(merged)]
    if finite.size == 0:
        vmin, vmax = -1.0, 1.0
    else:
        vmin = float(np.percentile(finite, 2))
        vmax = float(np.percentile(finite, 98))

    diff_finite = np.abs(diff[np.isfinite(diff)])
    dlim = float(np.percentile(diff_finite, 99)) if diff_finite.size else 1.0
    if dlim == 0.0:
        dlim = 1.0

    panels = [
        ("metrust", metrust, "viridis", vmin, vmax),
        ("MetPy", metpy, "viridis", vmin, vmax),
        ("metrust - MetPy", diff, "coolwarm", -dlim, dlim),
    ]
    for ax, (title, data, cmap, lo, hi) in zip(axes, panels, strict=True):
        im = ax.imshow(data, origin="upper", extent=extent, cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, shrink=0.9)

    fig.suptitle(name)
    fig.savefig(OUT_DIR / f"{name}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ensure_grib_inputs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    u500_da = load_cfgrib_var(FIELDS["u500"]["path"], "u")
    v500_da = load_cfgrib_var(FIELDS["v500"]["path"], "v")
    t850_da = load_cfgrib_var(FIELDS["t850"]["path"], "t")
    u850_da = load_cfgrib_var(FIELDS["u850"]["path"], "u")
    v850_da = load_cfgrib_var(FIELDS["v850"]["path"], "v")

    u500 = u500_da.values * units("m/s")
    v500 = v500_da.values * units("m/s")
    t850 = t850_da.values * units.kelvin
    u850 = u850_da.values * units("m/s")
    v850 = v850_da.values * units("m/s")

    lat = u500_da.coords["latitude"].values
    lon = u500_da.coords["longitude"].values
    lon2d, lat2d = np.meshgrid(lon, lat)
    dx, dy = mpcalc.lat_lon_grid_deltas(lon2d, lat2d)

    metpy_products = {
        "u500": u500.magnitude,
        "v500": v500.magnitude,
        "t850": t850.magnitude,
        "u850": u850.magnitude,
        "v850": v850.magnitude,
        "div500": np.asarray(mpcalc.divergence(u500, v500, dx=dx, dy=dy).magnitude, dtype=np.float64),
        "vort500": np.asarray(mpcalc.vorticity(u500, v500, dx=dx, dy=dy).magnitude, dtype=np.float64),
        "tadv850": np.asarray(mpcalc.advection(t850, u=u850, v=v850, dx=dx, dy=dy).magnitude, dtype=np.float64),
        "theta850": np.asarray(mpcalc.potential_temperature(850.0 * units.hPa, t850).magnitude, dtype=np.float64),
    }

    metrust_products = {
        "u500": u500.magnitude,
        "v500": v500.magnitude,
        "t850": t850.magnitude,
        "u850": u850.magnitude,
        "v850": v850.magnitude,
        "div500": np.asarray(mrcalc.divergence(u500, v500, dx=dx, dy=dy).magnitude, dtype=np.float64),
        "vort500": np.asarray(mrcalc.vorticity(u500, v500, dx=dx, dy=dy).magnitude, dtype=np.float64),
        "tadv850": np.asarray(mrcalc.advection(t850, u850, v850, dx=dx, dy=dy).magnitude, dtype=np.float64),
        "theta850": np.asarray(mrcalc.potential_temperature(850.0 * units.hPa, t850).magnitude, dtype=np.float64),
    }

    report: dict[str, dict[str, dict[str, float]]] = {}
    for name, metpy_values in metpy_products.items():
        metrust_values = metrust_products[name]
        report[name] = {
            "all_points": summary_stats(metpy_values, metrust_values, trim_edges=False),
            "core_points": summary_stats(metpy_values, metrust_values, trim_edges=True),
        }
        save_triptych(name, lon, lat, metrust_values, metpy_values)

    output = {
        "products": report,
    }
    (OUT_DIR / "comparison_summary.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
