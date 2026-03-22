from __future__ import annotations

import json
import subprocess
from pathlib import Path

import metpy.calc as mpcalc
import numpy as np
from metpy.interpolate import interpolate_1d
from metpy.units import units


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
OUT_DIR = EXAMPLES / "thermo_profile_verify"
CARGO = Path.home() / ".cargo" / "bin" / "cargo.exe"


def run(cmd: list[str]) -> None:
    resolved = [str(CARGO) if token == "cargo" else token for token in cmd]
    print(">", " ".join(resolved))
    subprocess.run(resolved, cwd=ROOT, check=True)


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


def scalar_magnitude(value, unit: str | None = None) -> float:
    if hasattr(value, "to") and unit is not None:
        value = value.to(unit)
    magnitude = getattr(value, "magnitude", value)
    return float(np.asarray(magnitude, dtype=np.float64).reshape(-1)[0])


def parcel_profile_500_c(pressure_hpa: np.ndarray, parcel_profile_k) -> float:
    value = interpolate_1d(500.0 * units.hPa, pressure_hpa * units.hPa, parcel_profile_k)
    return scalar_magnitude(value, "degC")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run(
        [
            "cargo",
            "run",
            "-p",
            "wx-cli",
            "--bin",
            "verify_thermo_profiles",
            "--",
            "--output-dir",
            str(OUT_DIR),
        ]
    )

    payload = json.loads((OUT_DIR / "wxforge_thermo_cases.json").read_text())
    wxforge_cases = {case["name"]: case for case in payload["cases"]}
    report: dict[str, dict[str, float]] = {}

    for name, case in wxforge_cases.items():
        pressure = np.asarray(case["profile"]["pressure_hpa"], dtype=np.float64) * units.hPa
        temperature = np.asarray(case["profile"]["temperature_c"], dtype=np.float64) * units.degC
        dewpoint = np.asarray(case["profile"]["dewpoint_c"], dtype=np.float64) * units.degC
        surface_p = float(case["surface"]["pressure_hpa"]) * units.hPa
        surface_t = float(case["surface"]["temperature_c"]) * units.degC
        surface_td = float(case["surface"]["dewpoint_c"]) * units.degC

        lcl_pressure, lcl_temperature = mpcalc.lcl(surface_p, surface_t, surface_td)
        parcel_profile = mpcalc.parcel_profile(pressure, surface_t, surface_td)

        metpy_products = {
            "lcl_pressure_hpa": scalar_magnitude(lcl_pressure, "hPa"),
            "lcl_temperature_c": scalar_magnitude(lcl_temperature, "degC"),
            "wet_bulb_temperature_c": scalar_magnitude(
                mpcalc.wet_bulb_temperature(surface_p, surface_t, surface_td),
                "degC",
            ),
            "wet_bulb_potential_temperature_k": scalar_magnitude(
                mpcalc.wet_bulb_potential_temperature(surface_p, surface_t, surface_td),
                "kelvin",
            ),
            "lifted_index_c": scalar_magnitude(
                mpcalc.lifted_index(pressure, temperature, parcel_profile),
                "delta_degC",
            ),
            "showalter_index_c": scalar_magnitude(
                mpcalc.showalter_index(pressure, temperature, dewpoint),
                "delta_degC",
            ),
            "k_index_c": scalar_magnitude(mpcalc.k_index(pressure, temperature, dewpoint), "degC"),
            "total_totals_c": scalar_magnitude(
                mpcalc.total_totals_index(pressure, temperature, dewpoint),
                "delta_degC",
            ),
            "precipitable_water_mm": scalar_magnitude(
                mpcalc.precipitable_water(pressure, dewpoint),
                "millimeter",
            ),
            "downdraft_cape_jkg": scalar_magnitude(
                mpcalc.downdraft_cape(pressure, temperature, dewpoint)[0],
                "J/kg",
            ),
            "parcel_profile_500_c": parcel_profile_500_c(pressure.magnitude, parcel_profile),
        }

        wxforge_products = case["products"]
        report[name] = {
            product_name: summary_stats(
                np.asarray([metpy_products[product_name]], dtype=np.float64),
                np.asarray([wxforge_products[product_name]], dtype=np.float64),
            )
            for product_name in metpy_products
        }

    output = {"cases": report}
    (OUT_DIR / "comparison_summary.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
