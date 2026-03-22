from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
DEFAULT_REPORT = EXAMPLES / "metpy_regression_report.json"


@dataclass(frozen=True)
class Check:
    path: str
    max_value: float
    description: str


@dataclass(frozen=True)
class Case:
    name: str
    description: str
    command: list[str]
    summary_path: Path
    checks: list[Check]


def gfs_map_checks() -> list[Check]:
    return pressure_map_checks(theta_mean_limit=1.0e-5)


def pressure_map_checks(theta_mean_limit: float) -> list[Check]:
    checks = [
        Check("coordinate_alignment.longitude.max_abs", 0.0, "longitude grid alignment"),
        Check("coordinate_alignment.latitude.max_abs", 0.0, "latitude grid alignment"),
    ]
    for field in ["u500", "v500", "t850", "u850", "v850"]:
        checks.append(Check(f"products.{field}.all_points.max_abs", 0.0, f"{field} exact field parity"))

    checks.extend(
        [
            Check("products.div500.core_points.mean_abs", 1.0e-8, "500-hPa divergence core mean abs"),
            Check("products.div500.core_points.max_abs", 1.0e-7, "500-hPa divergence core max abs"),
            Check("products.vort500.core_points.mean_abs", 1.0e-8, "500-hPa vorticity core mean abs"),
            Check("products.vort500.core_points.max_abs", 1.0e-7, "500-hPa vorticity core max abs"),
            Check("products.tadv850.core_points.mean_abs", 1.0e-7, "850-hPa temperature advection core mean abs"),
            Check("products.tadv850.core_points.max_abs", 1.0e-5, "850-hPa temperature advection core max abs"),
            Check("products.theta850.all_points.mean_abs", theta_mean_limit, "850-hPa theta mean abs"),
            Check("products.theta850.all_points.max_abs", 5.0e-5, "850-hPa theta max abs"),
        ]
    )
    return checks


def severe_checks(warm_case: bool) -> list[Check]:
    checks = [
        Check("products.sbcape.mean_abs", 0.1 if warm_case else 0.01, "surface CAPE mean abs"),
        Check("products.sbcape.max_abs", 0.5 if warm_case else 0.01, "surface CAPE max abs"),
        Check("products.sbcin.mean_abs", 0.01 if warm_case else 0.001, "surface CIN mean abs"),
        Check("products.sbcin.max_abs", 0.05 if warm_case else 0.01, "surface CIN max abs"),
        Check("products.h_lcl.mean_abs", 0.01, "surface LCL height mean abs"),
        Check("products.h_lcl.max_abs", 0.01, "surface LCL height max abs"),
        Check("products.mlcape.mean_abs", 0.1 if warm_case else 0.01, "mixed-layer CAPE mean abs"),
        Check("products.mlcape.max_abs", 0.5 if warm_case else 0.01, "mixed-layer CAPE max abs"),
        Check("products.mlcin.mean_abs", 0.01 if warm_case else 0.001, "mixed-layer CIN mean abs"),
        Check("products.mlcin.max_abs", 0.05 if warm_case else 0.01, "mixed-layer CIN max abs"),
        Check("products.mucape.mean_abs", 0.1 if warm_case else 0.01, "most-unstable CAPE mean abs"),
        Check("products.mucape.max_abs", 0.5 if warm_case else 0.01, "most-unstable CAPE max abs"),
        Check("products.mucin.mean_abs", 0.01 if warm_case else 0.001, "most-unstable CIN mean abs"),
        Check("products.mucin.max_abs", 0.05 if warm_case else 0.01, "most-unstable CIN max abs"),
        Check("products.mu_start_p.max_abs", 1.0e-4, "most-unstable parcel start pressure max abs"),
        Check("products.shear06.max_abs", 1.0e-4, "0-6 km bulk shear max abs"),
        Check("products.rm_u.max_abs", 1.0e-4, "Bunkers right-mover u max abs"),
        Check("products.rm_v.max_abs", 1.0e-4, "Bunkers right-mover v max abs"),
        Check("products.srh01.max_abs", 1.0e-3, "0-1 km SRH max abs"),
        Check("products.stp.max_abs", 1.0e-4, "STP max abs"),
        Check("products.li.mean_abs", 1.0e-3, "lifted index mean abs"),
        Check("products.li.max_abs", 1.0e-2, "lifted index max abs"),
        Check("products.showalter.mean_abs", 1.0e-3, "Showalter index mean abs"),
        Check("products.showalter.max_abs", 1.0e-2, "Showalter index max abs"),
        Check("products.k_index.mean_abs", 1.0e-3, "K-index mean abs"),
        Check("products.k_index.max_abs", 1.0e-2, "K-index max abs"),
        Check("products.total_totals.mean_abs", 1.0e-3, "total totals mean abs"),
        Check("products.total_totals.max_abs", 1.0e-2, "total totals max abs"),
        Check("products.pwat.mean_abs", 1.0e-2, "precipitable water mean abs"),
        Check("products.pwat.max_abs", 5.0e-2, "precipitable water max abs"),
    ]
    return checks


def decode_checks() -> list[Check]:
    return [
        Check("field.mean_abs", 0.0, "decoded field mean abs"),
        Check("field.rmse", 0.0, "decoded field rmse"),
        Check("field.max_abs", 0.0, "decoded field max abs"),
        Check("field.p99_abs", 0.0, "decoded field p99 abs"),
        Check("field.mean_bias", 0.0, "decoded field mean bias"),
    ]


def thermo_checks() -> list[Check]:
    checks = []
    for case_name in ["metpy_doc_profile", "plains_profile", "tropical_profile"]:
        checks.extend(
            [
                Check(f"cases.{case_name}.lcl_pressure_hpa.max_abs", 1.0e-9, f"{case_name} LCL pressure max abs"),
                Check(f"cases.{case_name}.lcl_temperature_c.max_abs", 1.0e-9, f"{case_name} LCL temperature max abs"),
                Check(f"cases.{case_name}.wet_bulb_temperature_c.max_abs", 1.0e-4, f"{case_name} wet-bulb temperature max abs"),
                Check(
                    f"cases.{case_name}.wet_bulb_potential_temperature_k.max_abs",
                    1.0e-9,
                    f"{case_name} wet-bulb potential temperature max abs",
                ),
                Check(f"cases.{case_name}.lifted_index_c.max_abs", 1.0e-3, f"{case_name} lifted index max abs"),
                Check(f"cases.{case_name}.showalter_index_c.max_abs", 1.0e-3, f"{case_name} showalter index max abs"),
                Check(f"cases.{case_name}.k_index_c.max_abs", 1.0e-9, f"{case_name} K-index max abs"),
                Check(f"cases.{case_name}.total_totals_c.max_abs", 1.0e-9, f"{case_name} total totals max abs"),
                Check(f"cases.{case_name}.precipitable_water_mm.max_abs", 5.0e-3, f"{case_name} PWAT max abs"),
                Check(f"cases.{case_name}.downdraft_cape_jkg.max_abs", 1.0e-2, f"{case_name} DCAPE max abs"),
                Check(f"cases.{case_name}.parcel_profile_500_c.max_abs", 1.0e-3, f"{case_name} parcel profile max abs"),
            ]
        )
    return checks


def build_cases() -> dict[str, Case]:
    python = sys.executable
    return {
        "gfs_maps": Case(
            name="gfs_maps",
            description="Large-area GFS analysis map parity against MetPy.",
            command=[python, "examples/compare_gfs_with_metpy.py", "--ensure-gfs-defaults"],
            summary_path=EXAMPLES / "gfs_metpy_verify" / "comparison_summary.json",
            checks=gfs_map_checks(),
        ),
        "gfs_maps_f006": Case(
            name="gfs_maps_f006",
            description="Large-area GFS +6h pressure-map parity against MetPy.",
            command=[
                python,
                "examples/compare_gfs_with_metpy.py",
                "--u500",
                "examples/verify_gfs_f006_ugrd500.grib2",
                "--v500",
                "examples/verify_gfs_f006_vgrd500.grib2",
                "--t850",
                "examples/verify_gfs_f006_tmp850.grib2",
                "--u850",
                "examples/verify_gfs_f006_ugrd850.grib2",
                "--v850",
                "examples/verify_gfs_f006_vgrd850.grib2",
                "--output-dir",
                "examples/gfs_metpy_verify_f006",
            ],
            summary_path=EXAMPLES / "gfs_metpy_verify_f006" / "comparison_summary.json",
            checks=pressure_map_checks(theta_mean_limit=1.0e-5),
        ),
        "gfs_maps_f012": Case(
            name="gfs_maps_f012",
            description="Large-area GFS +12h pressure-map parity against MetPy.",
            command=[
                python,
                "examples/compare_gfs_with_metpy.py",
                "--u500",
                "examples/verify_gfs_f012_ugrd500.grib2",
                "--v500",
                "examples/verify_gfs_f012_vgrd500.grib2",
                "--t850",
                "examples/verify_gfs_f012_tmp850.grib2",
                "--u850",
                "examples/verify_gfs_f012_ugrd850.grib2",
                "--v850",
                "examples/verify_gfs_f012_vgrd850.grib2",
                "--output-dir",
                "examples/gfs_metpy_verify_f012",
            ],
            summary_path=EXAMPLES / "gfs_metpy_verify_f012" / "comparison_summary.json",
            checks=pressure_map_checks(theta_mean_limit=1.0e-5),
        ),
        "era5_maps": Case(
            name="era5_maps",
            description="ERA5 pressure-level map parity against MetPy on a CONUS crop.",
            command=[
                python,
                "examples/compare_gfs_with_metpy.py",
                "--u500",
                "examples/verify_era5_ugrd500.grib",
                "--v500",
                "examples/verify_era5_vgrd500.grib",
                "--t850",
                "examples/verify_era5_tmp850.grib",
                "--u850",
                "examples/verify_era5_ugrd850.grib",
                "--v850",
                "examples/verify_era5_vgrd850.grib",
                "--output-dir",
                "examples/era5_metpy_verify",
            ],
            summary_path=EXAMPLES / "era5_metpy_verify" / "comparison_summary.json",
            checks=pressure_map_checks(theta_mean_limit=2.5e-5),
        ),
        "ecmwf_maps": Case(
            name="ecmwf_maps",
            description="ECMWF open-data pressure-level map parity against MetPy.",
            command=[
                python,
                "examples/compare_gfs_with_metpy.py",
                "--u500",
                "examples/verify_ecmwf_ugrd500.grib2",
                "--v500",
                "examples/verify_ecmwf_vgrd500.grib2",
                "--t850",
                "examples/verify_ecmwf_tmp850.grib2",
                "--u850",
                "examples/verify_ecmwf_ugrd850.grib2",
                "--v850",
                "examples/verify_ecmwf_vgrd850.grib2",
                "--output-dir",
                "examples/ecmwf_metpy_verify",
            ],
            summary_path=EXAMPLES / "ecmwf_metpy_verify" / "comparison_summary.json",
            checks=pressure_map_checks(theta_mean_limit=1.0e-5),
        ),
        "severe_plains": Case(
            name="severe_plains",
            description="Cool-season/plains severe profile parity against MetPy.",
            command=[
                python,
                "examples/compare_severe_profiles_with_metpy.py",
                "--data-dir",
                "examples/severe_gfs_verify",
                "--lat-min",
                "30",
                "--lat-max",
                "40",
                "--lon-min-360",
                "258",
                "--lon-max-360",
                "268",
                "--sample-stride",
                "4",
            ],
            summary_path=EXAMPLES / "severe_gfs_verify" / "metpy_compare" / "comparison_summary.json",
            checks=severe_checks(warm_case=False),
        ),
        "severe_gulf": Case(
            name="severe_gulf",
            description="Warm/moist Gulf severe profile parity against MetPy.",
            command=[
                python,
                "examples/compare_severe_profiles_with_metpy.py",
                "--data-dir",
                "examples/severe_gfs_verify_gulf",
                "--lat-min",
                "22",
                "--lat-max",
                "35",
                "--lon-min-360",
                "260",
                "--lon-max-360",
                "284",
                "--sample-stride",
                "5",
            ],
            summary_path=EXAMPLES / "severe_gfs_verify_gulf" / "metpy_compare" / "comparison_summary.json",
            checks=severe_checks(warm_case=True),
        ),
        "severe_southeast_f006": Case(
            name="severe_southeast_f006",
            description="Nocturnal Southeast GFS +6h severe-profile parity against MetPy.",
            command=[
                python,
                "examples/compare_severe_profiles_with_metpy.py",
                "--data-dir",
                "examples/severe_gfs_verify_f006",
                "--lat-min",
                "28",
                "--lat-max",
                "37",
                "--lon-min-360",
                "268",
                "--lon-max-360",
                "285",
                "--sample-stride",
                "6",
            ],
            summary_path=EXAMPLES / "severe_gfs_verify_f006" / "metpy_compare" / "comparison_summary.json",
            checks=severe_checks(warm_case=True),
        ),
        "severe_florida_elevated_f012": Case(
            name="severe_florida_elevated_f012",
            description="Elevated Florida GFS +12h severe-profile parity against MetPy.",
            command=[
                python,
                "examples/compare_severe_profiles_with_metpy.py",
                "--data-dir",
                "examples/severe_gfs_verify_f012",
                "--lat-min",
                "24",
                "--lat-max",
                "30",
                "--lon-min-360",
                "276",
                "--lon-max-360",
                "282",
                "--sample-stride",
                "4",
            ],
            summary_path=EXAMPLES / "severe_gfs_verify_f012" / "metpy_compare" / "comparison_summary.json",
            checks=severe_checks(warm_case=True),
        ),
        "hrrr_decode": Case(
            name="hrrr_decode",
            description="HRRR 2-m temperature decode parity against cfgrib.",
            command=[
                python,
                "examples/compare_field_decode.py",
                "--file",
                "examples/hrrr_2t_subset.grib2",
                "--var",
                "t2m",
                "--output-dir",
                "examples/hrrr_decode_verify",
            ],
            summary_path=EXAMPLES / "hrrr_decode_verify" / "comparison_summary.json",
            checks=decode_checks(),
        ),
        "rap_decode": Case(
            name="rap_decode",
            description="RAP 2-m temperature decode parity against cfgrib.",
            command=[
                python,
                "examples/compare_field_decode.py",
                "--file",
                "examples/rap_2t_subset.grib2",
                "--var",
                "t2m",
                "--output-dir",
                "examples/rap_decode_verify",
            ],
            summary_path=EXAMPLES / "rap_decode_verify" / "comparison_summary.json",
            checks=decode_checks(),
        ),
        "nam_decode": Case(
            name="nam_decode",
            description="NAM 2-m temperature decode parity against cfgrib.",
            command=[
                python,
                "examples/compare_field_decode.py",
                "--file",
                "examples/nam_2t_subset.grib2",
                "--var",
                "t2m",
                "--output-dir",
                "examples/nam_decode_verify",
            ],
            summary_path=EXAMPLES / "nam_decode_verify" / "comparison_summary.json",
            checks=decode_checks(),
        ),
        "ecmwf_decode": Case(
            name="ecmwf_decode",
            description="ECMWF open-data 2-m field decode parity against cfgrib.",
            command=[
                python,
                "examples/compare_field_decode.py",
                "--file",
                "examples/ecmwf_2t_subset.grib2",
                "--var",
                "mx2t3",
                "--output-dir",
                "examples/ecmwf_decode_verify",
            ],
            summary_path=EXAMPLES / "ecmwf_decode_verify" / "comparison_summary.json",
            checks=decode_checks(),
        ),
        "era5_decode": Case(
            name="era5_decode",
            description="ERA5 2-m temperature decode parity against cfgrib.",
            command=[
                python,
                "examples/compare_field_decode.py",
                "--file",
                "examples/era5_2t_subset.grib",
                "--var",
                "t2m",
                "--output-dir",
                "examples/era5_decode_verify",
            ],
            summary_path=EXAMPLES / "era5_decode_verify" / "comparison_summary.json",
            checks=decode_checks(),
        ),
        "thermo_profiles": Case(
            name="thermo_profiles",
            description="Standalone thermo/profile parity against MetPy on fixed sounding cases.",
            command=[python, "examples/compare_thermo_profiles_with_metpy.py"],
            summary_path=EXAMPLES / "thermo_profile_verify" / "comparison_summary.json",
            checks=thermo_checks(),
        ),
    }


def parse_args(case_names: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run thresholded wxforge-vs-MetPy regression cases.")
    parser.add_argument(
        "--case",
        action="append",
        choices=case_names,
        help="Run only the named case. Repeat to run multiple cases.",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Do not rerun the underlying comparison scripts; only evaluate existing summaries.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="Where to write the combined JSON report.",
    )
    return parser.parse_args()


def run_case(case: Case) -> None:
    print(">", " ".join(case.command))
    subprocess.run(case.command, cwd=ROOT, check=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve_path(document: dict[str, Any], dotted_path: str) -> Any:
    value: Any = document
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"missing path '{dotted_path}' at '{part}'")
        value = value[part]
    return value


def evaluate_case(case: Case, summary: dict[str, Any]) -> dict[str, Any]:
    checks_out = []
    failed = 0
    for check in case.checks:
        actual = float(resolve_path(summary, check.path))
        passed = actual <= check.max_value
        if not passed:
            failed += 1
        checks_out.append(
            {
                "path": check.path,
                "description": check.description,
                "actual": actual,
                "max_allowed": check.max_value,
                "passed": passed,
            }
        )
    return {
        "name": case.name,
        "description": case.description,
        "summary_path": str(case.summary_path.relative_to(ROOT)),
        "check_count": len(case.checks),
        "failed_checks": failed,
        "passed": failed == 0,
        "checks": checks_out,
    }


def main() -> int:
    cases = build_cases()
    args = parse_args(sorted(cases))
    selected_names = args.case or list(cases)

    case_reports = []
    for name in selected_names:
        case = cases[name]
        if not args.no_run:
            run_case(case)
        summary = load_json(case.summary_path)
        case_reports.append(evaluate_case(case, summary))

    total_checks = sum(case["check_count"] for case in case_reports)
    total_failed = sum(case["failed_checks"] for case in case_reports)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "cases": case_reports,
        "totals": {
            "case_count": len(case_reports),
            "check_count": total_checks,
            "failed_checks": total_failed,
            "passed": total_failed == 0,
        },
    }

    args.report.write_text(json.dumps(report, indent=2))

    for case in case_reports:
        status = "PASS" if case["passed"] else "FAIL"
        print(
            f"{status} {case['name']}: "
            f"{case['check_count'] - case['failed_checks']}/{case['check_count']} checks"
        )
        for check in case["checks"]:
            if not check["passed"]:
                print(
                    f"  {check['path']}: actual={check['actual']:.12g} "
                    f"limit={check['max_allowed']:.12g}"
                )

    print(f"wrote {args.report}")
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
