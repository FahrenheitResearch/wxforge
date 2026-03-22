from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(">", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare a single decoded wxtrain GRIB field against cfgrib.")
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--var", required=True, help="Data variable name as seen by cfgrib, e.g. t2m")
    parser.add_argument("--message", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def summary_stats(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
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


def save_triptych(name: str, reference: np.ndarray, candidate: np.ndarray, out_dir: Path) -> None:
    diff = candidate - reference
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    combined = np.concatenate([reference.ravel(), candidate.ravel()])
    finite = combined[np.isfinite(combined)]
    if finite.size == 0:
        vmin = -1.0
        vmax = 1.0
    else:
        vmin = np.percentile(finite, 2)
        vmax = np.percentile(finite, 98)
        if vmin == vmax:
            vmax = vmin + 1.0

    diff_finite = np.abs(diff[np.isfinite(diff)])
    dlim = np.percentile(diff_finite, 99) if diff_finite.size else 1.0
    if dlim == 0.0:
        dlim = 1.0

    panels = [
        ("wxtrain", candidate, "viridis", vmin, vmax),
        ("cfgrib", reference, "viridis", vmin, vmax),
        ("wxtrain - cfgrib", diff, "coolwarm", -dlim, dlim),
    ]
    for ax, (title, data, cmap, lo, hi) in zip(axes, panels, strict=True):
        im = ax.imshow(data, origin="upper", cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.9)

    fig.suptitle(name)
    fig.savefig(out_dir / f"{name}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run(
        [
            "cargo",
            "run",
            "-p",
            "wx-cli",
            "--bin",
            "dump_grib_field",
            "--",
            "--file",
            str(args.file),
            "--message",
            str(args.message),
            "--output-dir",
            str(args.output_dir),
        ]
    )

    ds = xr.open_dataset(args.file, engine="cfgrib")
    reference = ds[args.var].values.astype(np.float64)
    candidate = np.load(args.output_dir / "field.npy").astype(np.float64)

    if reference.shape != candidate.shape:
        raise RuntimeError(f"shape mismatch: cfgrib={reference.shape} wxtrain={candidate.shape}")

    report = {
        "file": str(args.file),
        "var": args.var,
        "message": args.message,
        "shape": list(reference.shape),
        "field": summary_stats(reference, candidate),
    }
    (args.output_dir / "comparison_summary.json").write_text(json.dumps(report, indent=2))
    save_triptych(args.var, reference, candidate, args.output_dir)
    print(json.dumps(report, indent=2))
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
