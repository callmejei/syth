"""One-off project setup. Run this in a CML Session before starting the Application.

    python bootstrap.py              # dependencies only
    python bootstrap.py --demo 3000  # also generate a demo dataset

Installs into the Session's Python environment, which CML persists with the
project, so the Application starts with the packages already present.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(*args: str) -> None:
    print(f"==> {' '.join(args)}")
    result = subprocess.run(args, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo",
        type=int,
        default=0,
        metavar="ROWS",
        help="also generate a demo dataset of this many customers",
    )
    args = parser.parse_args()

    run(sys.executable, "-m", "pip", "install", "--upgrade", "pip")
    run(sys.executable, "-m", "pip", "install", "-r", "requirements.txt")

    if args.demo:
        run(
            sys.executable,
            "-m",
            "scripts.make_demo_data",
            "--rows",
            str(args.demo),
            "--out",
            "data/demo",
        )

    static = ROOT / "static" / "index.html"
    if not static.is_file():
        print(
            "\nWARNING: static/ has no built UI. Build it on a machine with Node\n"
            "(python build_ui.py) and commit static/. The API still works without it."
        )

    print("\nSetup complete. Now create a CML Application with script: cml_app.py")


if __name__ == "__main__":
    main()
