from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
OUT_DIR = EXAMPLES / "ml_pipeline_verify"
REPORT_PATH = EXAMPLES / "ml_pipeline_verify_report.json"
CARGO = Path.home() / ".cargo" / "bin" / "cargo.exe"


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    resolved = [str(CARGO) if token == "cargo" else token for token in command]
    return subprocess.run(resolved, cwd=ROOT, capture_output=True, text=True, check=True)


def read_json_from_stdout(stdout: str) -> dict:
    lines = [line for line in stdout.splitlines() if line.strip()]
    return json.loads("\n".join(lines))


def verify_bundle(bundle_dir: Path) -> dict[str, object]:
    dataset_manifest = json.loads((bundle_dir / "dataset_manifest.json").read_text())
    sample_manifest = json.loads((bundle_dir / "sample_manifest.json").read_text())
    npy_files = sorted(path.name for path in bundle_dir.glob("*.npy"))
    png_files = sorted(path.name for path in bundle_dir.glob("*.png"))
    return {
        "dataset_name": dataset_manifest["dataset_name"],
        "format": dataset_manifest["format"],
        "channel_count": sample_manifest["channel_count"],
        "npy_files": npy_files,
        "png_files": png_files,
        "passed": bool(npy_files) and sample_manifest["channel_count"] == len(sample_manifest["channels"]),
    }


def verify_dataset(dataset_dir: Path) -> dict[str, object]:
    dataset_manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text())
    build_manifest = json.loads((dataset_dir / "dataset_build_manifest.json").read_text())
    sample_dirs = sorted(
        path.name for path in dataset_dir.iterdir() if path.is_dir() and path.name != "shards"
    )
    shard_files = sorted(path.name for path in (dataset_dir / "shards").glob("*.json"))
    tar_shards = sorted(path.name for path in (dataset_dir / "shards").glob("*.tar"))
    jsonl_shards = sorted(path.name for path in (dataset_dir / "shards").glob("*.jsonl"))
    parquet_shards = sorted(path.name for path in (dataset_dir / "shards").glob("*.parquet"))
    return {
        "dataset_name": dataset_manifest["dataset_name"],
        "format": dataset_manifest["format"],
        "sample_count": build_manifest["sample_count"],
        "total_channel_count": build_manifest["total_channel_count"],
        "shard_count": build_manifest["shard_count"],
        "split_counts": build_manifest["split_counts"],
        "sample_dirs": sample_dirs,
        "shard_files": shard_files,
        "tar_shards": tar_shards,
        "jsonl_shards": jsonl_shards,
        "parquet_shards": parquet_shards,
        "passed": (
            build_manifest["sample_count"] == len(build_manifest["samples"]) == len(sample_dirs)
            and build_manifest["shard_count"] == len(shard_files)
            and (
                dataset_manifest["format"] != "WebDataset"
                or build_manifest["shard_count"] == len(tar_shards)
            )
            and (
                dataset_manifest["format"] != "Jsonl"
                or build_manifest["shard_count"] == len(jsonl_shards)
            )
            and (
                dataset_manifest["format"] != "Parquet"
                or build_manifest["shard_count"] == len(parquet_shards)
            )
            and sum(build_manifest["split_counts"].values()) == build_manifest["sample_count"]
        ),
    }


def verify_agent_job_plan(plan: dict[str, object], *, expect_fetch: bool) -> dict[str, object]:
    dataset_build_request = plan.get("dataset_build_request")
    fetch_plan = plan.get("fetch_plan")
    model_recipe = plan.get("model_recipe", {})
    training = plan.get("training", {})
    spec = training.get("spec", {})
    warnings = plan.get("warnings", [])
    return {
        "job_name": plan.get("job_name"),
        "dataset_name": spec.get("dataset_name"),
        "channel_count": len(spec.get("channels", [])),
        "label_count": len(spec.get("labels", [])),
        "trainer_family": model_recipe.get("trainer_family"),
        "recommended_format": model_recipe.get("recommended_format"),
        "has_dataset_build_request": dataset_build_request is not None,
        "has_fetch_plan": fetch_plan is not None,
        "warning_count": len(warnings),
        "passed": (
            bool(spec.get("dataset_name"))
            and len(spec.get("channels", [])) > 0
            and model_recipe.get("trainer_family") is not None
            and (fetch_plan is not None) == expect_fetch
            and (dataset_build_request is None) == expect_fetch
        ),
    }


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline_plan = read_json_from_stdout(
        run(["cargo", "run", "-p", "wx-cli", "--bin", "wxtrain", "--", "train", "plan", "--preset", "baseline"]).stdout
    )
    radar_plan = read_json_from_stdout(
        run(["cargo", "run", "-p", "wx-cli", "--bin", "wxtrain", "--", "train", "plan", "--preset", "radar"]).stdout
    )

    sample_bundle = OUT_DIR / "sample_bundle"
    era5_bundle = OUT_DIR / "era5_bundle"
    batch_dataset = OUT_DIR / "batch_dataset"
    batch_jsonl_dataset = OUT_DIR / "batch_jsonl_dataset"
    batch_parquet_dataset = OUT_DIR / "batch_parquet_dataset"
    agent_job_init = OUT_DIR / "agent_job_init.json"
    agent_job_build = OUT_DIR / "agent_job_build"
    run(
        [
            "cargo",
            "run",
            "-p",
            "wx-cli",
            "--bin",
            "wxtrain",
            "--",
            "train",
            "build-grib-sample",
            "--file",
            "examples/sample.grib2",
            "--output-dir",
            str(sample_bundle),
            "--colormap",
            "heat",
        ]
    )
    run(
        [
            "cargo",
            "run",
            "-p",
            "wx-cli",
            "--bin",
            "wxtrain",
            "--",
            "train",
            "build-grib-dataset",
            "--manifest",
            "examples/sample_dataset_manifest_parquet.json",
            "--output-dir",
            str(batch_parquet_dataset),
            "--colormap",
            "heat",
        ]
    )
    run(
        [
            "cargo",
            "run",
            "-p",
            "wx-cli",
            "--bin",
            "wxtrain",
            "--",
            "train",
            "build-grib-dataset",
            "--manifest",
            "examples/sample_dataset_manifest_jsonl.json",
            "--output-dir",
            str(batch_jsonl_dataset),
            "--colormap",
            "heat",
        ]
    )
    run(
        [
            "cargo",
            "run",
            "-p",
            "wx-cli",
            "--bin",
            "wxtrain",
            "--",
            "train",
            "build-grib-dataset",
            "--manifest",
            "examples/sample_dataset_manifest.json",
            "--output-dir",
            str(batch_dataset),
            "--colormap",
            "heat",
        ]
    )
    run(
        [
            "cargo",
            "run",
            "-p",
            "wx-cli",
            "--bin",
            "wxtrain",
            "--",
            "train",
            "build-grib-sample",
            "--file",
            "examples/era5_2t_subset.grib",
            "--output-dir",
            str(era5_bundle),
            "--messages",
            "1",
            "--colormap",
            "heat",
        ]
    )
    run(
        [
            "cargo",
            "run",
            "-p",
            "wx-cli",
            "--bin",
            "wxtrain",
            "--",
            "train",
            "job-init",
            "--output",
            str(agent_job_init),
            "--architecture",
            "swin-transformer",
            "--task",
            "forecasting",
            "--dataset-name",
            "agent_smoke_dataset",
        ]
    )

    classical_agent_plan = read_json_from_stdout(
        run(
            [
                "cargo",
                "run",
                "-p",
                "wx-cli",
                "--bin",
                "wxtrain",
                "--",
                "train",
                "job-plan",
                "--spec",
                "examples/agent_job_classical.json",
            ]
        ).stdout
    )
    swin_agent_plan = read_json_from_stdout(
        run(
            [
                "cargo",
                "run",
                "-p",
                "wx-cli",
                "--bin",
                "wxtrain",
                "--",
                "train",
                "job-plan",
                "--spec",
                "examples/agent_job_swin.json",
            ]
        ).stdout
    )
    run(
        [
            "cargo",
            "run",
            "-p",
            "wx-cli",
            "--bin",
            "wxtrain",
            "--",
            "train",
            "job-build",
            "--spec",
            "examples/agent_job_classical.json",
            "--output-dir",
            str(agent_job_build),
            "--colormap",
            "heat",
        ]
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "plans": {
            "baseline": {
                "dataset_name": baseline_plan["dataset_name"],
                "format": baseline_plan["format"],
                "channel_count": len(baseline_plan["channels"]),
                "label_count": len(baseline_plan["labels"]),
                "passed": baseline_plan["dataset_name"] == "baseline-weather-dataset",
            },
            "radar": {
                "dataset_name": radar_plan["dataset_name"],
                "format": radar_plan["format"],
                "channel_count": len(radar_plan["channels"]),
                "label_count": len(radar_plan["labels"]),
                "passed": radar_plan["dataset_name"] == "radar-supervised-weather-dataset",
            },
        },
        "bundles": {
            "sample": verify_bundle(sample_bundle),
            "era5": verify_bundle(era5_bundle),
        },
        "datasets": {
            "batch": verify_dataset(batch_dataset),
            "batch_jsonl": verify_dataset(batch_jsonl_dataset),
            "batch_parquet": verify_dataset(batch_parquet_dataset),
        },
        "agent_jobs": {
            "job_init": {
                "path": str(agent_job_init.relative_to(ROOT)),
                "dataset_name": json.loads(agent_job_init.read_text())["dataset_name"],
                "passed": agent_job_init.exists(),
            },
            "classical_plan": verify_agent_job_plan(classical_agent_plan, expect_fetch=False),
            "swin_plan": verify_agent_job_plan(swin_agent_plan, expect_fetch=True),
            "classical_build": {
                **verify_dataset(agent_job_build),
                "planned_channel_count": len(
                    classical_agent_plan["training"]["spec"]["channels"]
                ),
                "job_plan_exists": (agent_job_build / "job_plan.json").exists(),
                "model_recipe_exists": (agent_job_build / "model_recipe.json").exists(),
                "dataset_request_exists": (agent_job_build / "dataset_request.json").exists(),
                "passed": (
                    verify_dataset(agent_job_build)["passed"]
                    and verify_dataset(agent_job_build)["total_channel_count"]
                    == len(classical_agent_plan["training"]["spec"]["channels"])
                    and (agent_job_build / "job_plan.json").exists()
                    and (agent_job_build / "model_recipe.json").exists()
                    and (agent_job_build / "dataset_request.json").exists()
                ),
            },
        },
    }
    report["passed"] = (
        all(section["passed"] for section in report["plans"].values())
        and all(section["passed"] for section in report["bundles"].values())
        and all(section["passed"] for section in report["datasets"].values())
        and all(section["passed"] for section in report["agent_jobs"].values())
    )
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
