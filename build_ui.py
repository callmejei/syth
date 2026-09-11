"""Build the React bundle into static/.

Run this on a machine that has Node, then commit static/ with the project.
CML runtimes generally have no Node, so the Application serves a bundle that
was built ahead of time.

    python build_ui.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

FRONTEND = Path(__file__).resolve().parent / "frontend"


def main() -> int:
    npm = shutil.which("npm")
    if npm is None:
        print("npm not found. Build static/ on a machine with Node and copy it here.")
        return 1

    lockfile = FRONTEND / "package-lock.json"
    install = ["ci"] if lockfile.is_file() else ["install"]

    for args in (install, ["run", "build"]):
        print(f"==> npm {' '.join(args)}")
        result = subprocess.run([npm, *args], cwd=FRONTEND)
        if result.returncode != 0:
            return result.returncode

    print("\nBuilt into static/. Commit it so the CML Application can serve it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
