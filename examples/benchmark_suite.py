from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
REPORT_PATH = EXAMPLES / "benchmark_report.json"
CARGO = Path.home() / ".cargo" / "bin" / "cargo.exe"


def run_timed(command: list[str]) -> dict[str, object]:
    resolved = [str(CARGO) if token == "cargo" else token for token in command]
    started = time.perf_counter()
    completed = subprocess.run(resolved, cwd=ROOT, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    return {
        "command": resolved,
        "elapsed_seconds": elapsed,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
        "passed": completed.returncode == 0,
    }


def main() -> None:
    cases = {
        "decode_gfs_tmp850": [
            "cargo",
            "run",
            "-p",
            "wx-cli",
            "--bin",
            "wxforge",
            "--",
            "decode-grib",
            "--file",
            "examples/verify_gfs_tmp850.grib2",
            "--message",
            "1",
            "--limit",
            "8",
        ],
        "verify_gfs_maps": [
            "cargo",
            "run",
            "-p",
            "wx-cli",
            "--bin",
            "verify_gfs_maps",
            "--",
            "--u500",
            "examples/verify_gfs_ugrd500.grib2",
            "--v500",
            "examples/verify_gfs_vgrd500.grib2",
            "--t850",
            "examples/verify_gfs_tmp850.grib2",
            "--u850",
            "examples/verify_gfs_ugrd850.grib2",
            "--v850",
            "examples/verify_gfs_vgrd850.grib2",
            "--latitude-npy",
            "examples/gfs_metpy_verify/latitude_input.npy",
            "--longitude-npy",
            "examples/gfs_metpy_verify/longitude_input.npy",
            "--output-dir",
            "examples/benchmark_gfs_maps",
        ],
        "verify_severe_profiles": [
            "cargo",
            "run",
            "-p",
            "wx-cli",
            "--bin",
            "verify_severe_profiles",
            "--",
            "--tmp",
            "examples/severe_gfs_verify/gfs_tmp_pressure.grib2",
            "--rh",
            "examples/severe_gfs_verify/gfs_rh_pressure.grib2",
            "--ugrd",
            "examples/severe_gfs_verify/gfs_ugrd_pressure.grib2",
            "--vgrd",
            "examples/severe_gfs_verify/gfs_vgrd_pressure.grib2",
            "--hgt",
            "examples/severe_gfs_verify/gfs_hgt_pressure.grib2",
            "--pres",
            "examples/severe_gfs_verify/gfs_pres_pressure.grib2",
            "--output-dir",
            "examples/benchmark_severe_profiles",
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
        "build_sample_bundle": [
            "cargo",
            "run",
            "-p",
            "wx-cli",
            "--bin",
            "wxforge",
            "--",
            "train",
            "build-grib-sample",
            "--file",
            "examples/sample.grib2",
            "--output-dir",
            "examples/benchmark_sample_bundle",
            "--colormap",
            "heat",
        ],
        "build_sample_dataset": [
            "cargo",
            "run",
            "-p",
            "wx-cli",
            "--bin",
            "wxforge",
            "--",
            "train",
            "build-grib-dataset",
            "--manifest",
            "examples/sample_dataset_manifest.json",
            "--output-dir",
            "examples/benchmark_batch_dataset",
            "--colormap",
            "heat",
        ],
    }

    results = {name: run_timed(command) for name, command in cases.items()}
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cases": results,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
