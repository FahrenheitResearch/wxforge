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


def run(cmd: list[str]) -> None:
    print(">", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


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


def save_triptych(
    name: str, lon: np.ndarray, lat: np.ndarray, wxtrain: np.ndarray, metpy: np.ndarray, out_dir: Path
) -> None:
    diff = wxtrain - metpy
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]

    combined = np.concatenate([wxtrain.ravel(), metpy.ravel()])
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
        ("wxtrain", wxtrain, "viridis", vmin, vmax),
        ("MetPy", metpy, "viridis", vmin, vmax),
        ("wxtrain - MetPy", diff, "coolwarm", -dlim, dlim),
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


def load_cfgrib(path: Path, filter_by_keys: dict[str, object]) -> xr.Dataset:
    return xr.open_dataset(path, engine="cfgrib", backend_kwargs={"filter_by_keys": filter_by_keys})


def scalar_magnitude(value, unit: str | None = None) -> float:
    if hasattr(value, "to") and unit is not None:
        value = value.to(unit)
    magnitude = getattr(value, "magnitude", value)
    array = np.asarray(magnitude, dtype=np.float64)
    if array.size == 0:
        return float("nan")
    return float(array.reshape(-1)[0])


def interp_height_at_pressure(target_hpa: float, pressure_hpa: np.ndarray, height_m: np.ndarray) -> float:
    if pressure_hpa.size == 0:
        return float("nan")
    if target_hpa >= pressure_hpa[0]:
        return float(height_m[0])
    if target_hpa <= pressure_hpa[-1]:
        return float(height_m[-1])
    for idx in range(1, pressure_hpa.size):
        if pressure_hpa[idx] <= target_hpa:
            log_target = np.log(target_hpa)
            log_p0 = np.log(pressure_hpa[idx - 1])
            log_p1 = np.log(pressure_hpa[idx])
            if abs(log_p1 - log_p0) < 1.0e-12:
                return float(height_m[idx - 1])
            frac = (log_target - log_p0) / (log_p1 - log_p0)
            return float(height_m[idx - 1] + frac * (height_m[idx] - height_m[idx - 1]))
    return float(height_m[-1])


def build_paths(data_dir: Path) -> tuple[Path, Path, dict[str, Path]]:
    wxtrain_dir = data_dir / "wxtrain_output"
    out_dir = data_dir / "metpy_compare"
    files = {
        "tmp": data_dir / "gfs_tmp_pressure.grib2",
        "rh": data_dir / "gfs_rh_pressure.grib2",
        "ugrd": data_dir / "gfs_ugrd_pressure.grib2",
        "vgrd": data_dir / "gfs_vgrd_pressure.grib2",
        "hgt": data_dir / "gfs_hgt_pressure.grib2",
        "pres": data_dir / "gfs_pres_pressure.grib2",
    }
    return wxtrain_dir, out_dir, files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare wxtrain severe diagnostics with MetPy.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=EXAMPLES / "severe_gfs_verify",
        help="Directory containing the GRIB subset files.",
    )
    parser.add_argument("--lat-min", type=float, default=30.0)
    parser.add_argument("--lat-max", type=float, default=40.0)
    parser.add_argument("--lon-min-360", type=float, default=258.0)
    parser.add_argument("--lon-max-360", type=float, default=268.0)
    parser.add_argument("--sample-stride", type=int, default=4)
    return parser.parse_args()


def build_wxtrain_outputs(
    data_dir: Path,
    wxtrain_dir: Path,
    files: dict[str, Path],
    lat_min: float,
    lat_max: float,
    lon_min_360: float,
    lon_max_360: float,
    sample_stride: int,
) -> None:
    wxtrain_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            "cargo",
            "run",
            "-p",
            "wx-cli",
            "--bin",
            "verify_severe_profiles",
            "--",
            "--tmp",
            str(files["tmp"]),
            "--rh",
            str(files["rh"]),
            "--ugrd",
            str(files["ugrd"]),
            "--vgrd",
            str(files["vgrd"]),
            "--hgt",
            str(files["hgt"]),
            "--pres",
            str(files["pres"]),
            "--output-dir",
            str(wxtrain_dir),
            "--lat-min",
            str(lat_min),
            "--lat-max",
            str(lat_max),
            "--lon-min-360",
            str(lon_min_360),
            "--lon-max-360",
            str(lon_max_360),
            "--sample-stride",
            str(sample_stride),
        ]
    )


def map_crop_indices(wx_lat: np.ndarray, wx_lon: np.ndarray, lat_axis: np.ndarray, lon_axis: np.ndarray) -> tuple[list[int], list[int]]:
    y_indices = []
    x_indices = []
    for lat in wx_lat[:, 0]:
        idx = int(np.argmin(np.abs(lat_axis - lat)))
        if abs(lat_axis[idx] - lat) > 1.0e-6:
            raise RuntimeError(f"latitude lookup failed for {lat}")
        y_indices.append(idx)
    lon_lookup = np.where(wx_lon[0, :] < 0.0, wx_lon[0, :] + 360.0, wx_lon[0, :])
    for lon in lon_lookup:
        idx = int(np.argmin(np.abs(lon_axis - lon)))
        if abs(lon_axis[idx] - lon) > 1.0e-6:
            raise RuntimeError(f"longitude lookup failed for {lon}")
        x_indices.append(idx)
    return y_indices, x_indices


def main() -> None:
    args = parse_args()
    wxtrain_dir, out_dir, files = build_paths(args.data_dir)
    build_wxtrain_outputs(
        args.data_dir,
        wxtrain_dir,
        files,
        args.lat_min,
        args.lat_max,
        args.lon_min_360,
        args.lon_max_360,
        args.sample_stride,
    )

    ds_t = load_cfgrib(files["tmp"], {"typeOfLevel": "isobaricInhPa"})
    ds_rh = load_cfgrib(files["rh"], {"typeOfLevel": "isobaricInhPa"})
    ds_u = load_cfgrib(files["ugrd"], {"typeOfLevel": "isobaricInhPa"})
    ds_v = load_cfgrib(files["vgrd"], {"typeOfLevel": "isobaricInhPa"})
    ds_h = load_cfgrib(files["hgt"], {"typeOfLevel": "isobaricInhPa"})
    ds_t2m = load_cfgrib(files["tmp"], {"typeOfLevel": "heightAboveGround", "level": 2})
    ds_r2m = load_cfgrib(files["rh"], {"typeOfLevel": "heightAboveGround", "level": 2})
    ds_u10 = load_cfgrib(files["ugrd"], {"typeOfLevel": "heightAboveGround", "level": 10})
    ds_v10 = load_cfgrib(files["vgrd"], {"typeOfLevel": "heightAboveGround", "level": 10})
    ds_orog = load_cfgrib(files["hgt"], {"typeOfLevel": "surface"})
    ds_sp = load_cfgrib(files["pres"], {"typeOfLevel": "surface"})

    wx_lat = np.load(wxtrain_dir / "latitude.npy").astype(np.float64)
    wx_lon = np.load(wxtrain_dir / "longitude.npy").astype(np.float64)
    lat_axis = ds_t.latitude.values.astype(np.float64)
    lon_axis = ds_t.longitude.values.astype(np.float64)
    y_indices, x_indices = map_crop_indices(wx_lat, wx_lon, lat_axis, lon_axis)

    pressure_hpa = ds_t.isobaricInhPa.values.astype(np.float64)
    pressure_mask = (pressure_hpa >= 100.0) & (pressure_hpa <= 1000.0)
    pressure_hpa = pressure_hpa[pressure_mask]

    t3d = ds_t["t"].values.astype(np.float64)[pressure_mask]
    rh3d = ds_rh["r"].values.astype(np.float64)[pressure_mask]
    u3d = ds_u["u"].values.astype(np.float64)[pressure_mask]
    v3d = ds_v["v"].values.astype(np.float64)[pressure_mask]
    h3d = ds_h["gh"].values.astype(np.float64)[pressure_mask]

    t2m = ds_t2m["t2m"].values.astype(np.float64)
    r2m = ds_r2m["r2"].values.astype(np.float64)
    u10 = ds_u10["u10"].values.astype(np.float64)
    v10 = ds_v10["v10"].values.astype(np.float64)
    orog = ds_orog["orog"].values.astype(np.float64)
    sp = ds_sp["sp"].values.astype(np.float64)

    fields = [
        "sbcape",
        "sbcin",
        "h_lcl",
        "mlcape",
        "mlcin",
        "mucape",
        "mucin",
        "mu_start_p",
        "shear06",
        "rm_u",
        "rm_v",
        "srh01",
        "stp",
        "li",
        "showalter",
        "k_index",
        "total_totals",
        "pwat",
    ]
    wxtrain_products = {
        name: np.load(wxtrain_dir / f"{name}.npy").astype(np.float64)
        for name in fields
    }
    metpy_products = {name: np.full_like(wx_lat, np.nan, dtype=np.float64) for name in fields}

    for oy, iy in enumerate(y_indices):
        for ox, ix in enumerate(x_indices):
            psfc_hpa = sp[iy, ix] / 100.0
            if not np.isfinite(psfc_hpa) or psfc_hpa < 800.0 or psfc_hpa > 1100.0:
                continue

            t2m_q = t2m[iy, ix] * units.kelvin
            rh2m = (r2m[iy, ix] / 100.0) * units.dimensionless
            td2m_q = mpcalc.dewpoint_from_relative_humidity(t2m_q, rh2m)
            u10_q = u10[iy, ix] * units("m/s")
            v10_q = v10[iy, ix] * units("m/s")
            surface_height = orog[iy, ix]

            selected = []
            for level_idx, p_hpa in enumerate(pressure_hpa):
                if p_hpa >= psfc_hpa - 0.1:
                    continue
                temp_k = t3d[level_idx, iy, ix]
                rh_pct = rh3d[level_idx, iy, ix]
                wind_u = u3d[level_idx, iy, ix]
                wind_v = v3d[level_idx, iy, ix]
                height_agl = h3d[level_idx, iy, ix] - surface_height
                if (
                    not np.isfinite(temp_k)
                    or not np.isfinite(rh_pct)
                    or not np.isfinite(wind_u)
                    or not np.isfinite(wind_v)
                    or not np.isfinite(height_agl)
                    or height_agl < 0.0
                ):
                    continue
                temp_q = temp_k * units.kelvin
                td_q = mpcalc.dewpoint_from_relative_humidity(
                    temp_q, (rh_pct / 100.0) * units.dimensionless
                )
                selected.append(
                    (
                        p_hpa * units.hPa,
                        temp_q,
                        td_q,
                        height_agl * units.meter,
                        wind_u * units("m/s"),
                        wind_v * units("m/s"),
                    )
                )

            if len(selected) < 6:
                continue

            p_full = units.Quantity(
                [psfc_hpa, *[entry[0].magnitude for entry in selected]], "hPa"
            )
            t_full = units.Quantity(
                [t2m_q.magnitude, *[entry[1].magnitude for entry in selected]], "kelvin"
            )
            td_full = units.Quantity(
                [td2m_q.to("kelvin").magnitude, *[entry[2].to("kelvin").magnitude for entry in selected]],
                "kelvin",
            )
            height_full = units.Quantity(
                [0.0, *[entry[3].magnitude for entry in selected]], "meter"
            )
            u_full = units.Quantity(
                [u10_q.magnitude, *[entry[4].magnitude for entry in selected]], "m/s"
            )
            v_full = units.Quantity(
                [v10_q.magnitude, *[entry[5].magnitude for entry in selected]], "m/s"
            )

            try:
                sbcape, sbcin = mpcalc.surface_based_cape_cin(p_full, t_full, td_full)
                mlcape, mlcin = mpcalc.mixed_layer_cape_cin(
                    p_full, t_full, td_full, depth=100.0 * units.hPa
                )
                mucape, mucin = mpcalc.most_unstable_cape_cin(
                    p_full, t_full, td_full, depth=300.0 * units.hPa
                )
                mu_p, _, _, _ = mpcalc.most_unstable_parcel(
                    p_full, t_full, td_full, depth=300.0 * units.hPa
                )
                shear_u, shear_v = mpcalc.bulk_shear(
                    p_full, u_full, v_full, height=height_full, depth=6.0 * units.km
                )
                shear06 = np.hypot(shear_u.to("m/s").magnitude, shear_v.to("m/s").magnitude) * units("m/s")
                rm, _, _ = mpcalc.bunkers_storm_motion(p_full, u_full, v_full, height_full)
                srh_pos, srh_neg, srh_tot = mpcalc.storm_relative_helicity(
                    height_full,
                    u_full,
                    v_full,
                    depth=1.0 * units.km,
                    storm_u=rm[0],
                    storm_v=rm[1],
                )
                lcl_pressure, _ = mpcalc.lcl(p_full[0], t_full[0], td_full[0])
                lcl_height_m = interp_height_at_pressure(
                    lcl_pressure.to("hPa").magnitude,
                    p_full.to("hPa").magnitude,
                    height_full.to("meter").magnitude,
                )
                stp = mpcalc.significant_tornado(
                    sbcape,
                    lcl_height_m * units.meter,
                    srh_tot,
                    shear06,
                )
                parcel_profile = mpcalc.parcel_profile(p_full, t_full[0], td_full[0])
                li = mpcalc.lifted_index(p_full, t_full, parcel_profile)
                showalter = mpcalc.showalter_index(p_full, t_full, td_full)
                k_index = mpcalc.k_index(p_full, t_full, td_full)
                total_totals = mpcalc.total_totals_index(p_full, t_full, td_full)
                pwat = mpcalc.precipitable_water(p_full, td_full)
            except Exception:
                continue

            metpy_products["sbcape"][oy, ox] = scalar_magnitude(sbcape, "J/kg")
            metpy_products["sbcin"][oy, ox] = scalar_magnitude(sbcin, "J/kg")
            metpy_products["h_lcl"][oy, ox] = lcl_height_m
            metpy_products["mlcape"][oy, ox] = scalar_magnitude(mlcape, "J/kg")
            metpy_products["mlcin"][oy, ox] = scalar_magnitude(mlcin, "J/kg")
            metpy_products["mucape"][oy, ox] = scalar_magnitude(mucape, "J/kg")
            metpy_products["mucin"][oy, ox] = scalar_magnitude(mucin, "J/kg")
            metpy_products["mu_start_p"][oy, ox] = scalar_magnitude(mu_p, "hPa")
            metpy_products["shear06"][oy, ox] = scalar_magnitude(shear06, "m/s")
            metpy_products["rm_u"][oy, ox] = scalar_magnitude(rm[0], "m/s")
            metpy_products["rm_v"][oy, ox] = scalar_magnitude(rm[1], "m/s")
            metpy_products["srh01"][oy, ox] = scalar_magnitude(srh_tot, "m^2/s^2")
            metpy_products["stp"][oy, ox] = scalar_magnitude(stp)
            metpy_products["li"][oy, ox] = scalar_magnitude(li)
            metpy_products["showalter"][oy, ox] = scalar_magnitude(showalter)
            metpy_products["k_index"][oy, ox] = scalar_magnitude(k_index)
            metpy_products["total_totals"][oy, ox] = scalar_magnitude(total_totals)
            metpy_products["pwat"][oy, ox] = scalar_magnitude(pwat, "millimeter")

    report = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in fields:
        report[name] = summary_stats(metpy_products[name], wxtrain_products[name])
        save_triptych(name, wx_lon, wx_lat, wxtrain_products[name], metpy_products[name], out_dir)

    output = {"products": report}
    (out_dir / "comparison_summary.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
