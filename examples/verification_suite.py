from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
REPORT_PATH = EXAMPLES / "verification_suite_report.json"


SECTIONS = {
    "metpy": [sys.executable, "examples/metpy_regression_suite.py", "--no-run"],
    "thermo": [sys.executable, "examples/compare_thermo_profiles_with_metpy.py"],
    "bench": [sys.executable, "examples/benchmark_suite.py"],
    "ml": [sys.executable, "examples/verify_ml_pipeline.py"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full local wxforge verification stack.")
    parser.add_argument(
        "--section",
        action="append",
        choices=sorted(SECTIONS.keys()),
        help="Run only selected section(s). Defaults to all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sections = args.section or list(SECTIONS)
    results = {}
    for name in sections:
        command = SECTIONS[name]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        results[name] = {
            "command": command,
            "returncode": completed.returncode,
            "passed": completed.returncode == 0,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sections": results,
        "passed": all(result["passed"] for result in results.values()),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
