"""Entry point for the CML Application.

Create the Application in CML with this file as the script. CML sets
CDSW_APP_PORT and expects the process to bind it on 0.0.0.0 and stay in the
foreground.

    Name    Synthforge
    Script  cml_app.py
    Kernel  Python 3.10+
"""

import os

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("CDSW_APP_PORT", "8100"))
    # No --reload and a single worker: reload spawns a child CML cannot see, and
    # a second worker would mean two thread pools writing the same SQLite file.
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, workers=1, log_level="info")
