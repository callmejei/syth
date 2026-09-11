# Synthforge — CML build

The same platform as `../poc`, repackaged to run as a **single Cloudera Machine
Learning Application**: one process, one port, no containers to orchestrate.

`../poc` is unchanged and still runs under Docker Compose. This is a parallel
copy, not a replacement.

---

## Why it is a separate build

A CML Application is one container running one process on one port
(`$CDSW_APP_PORT`). There is no Docker socket, no second container to put a
worker in, and usually no root to install Postgres or Redis. So the four pieces
of the Compose stack that assumed external services were replaced:

| Compose stack | CML build |
|---|---|
| Postgres | SQLite at `data/synthforge.db` (WAL mode) |
| Redis + Celery worker container | `ThreadPoolExecutor` inside the app process |
| MinIO / S3 mirror | Dropped — artifacts already live on the project filesystem |
| Separate Vite dev server on :5173 | React bundle prebuilt into `static/`, served by FastAPI |
| Atlas container | Off by default; imported catalog or bundled fixture |

Everything else — the SPN engine, the Atlas policy engine, the schema loader,
the API, the whole UI — is byte-identical to the Docker build.

The API's own code was **not** rewritten for this: `app/worker.py` keeps the
`train_task.delay(job_id)` call shape Celery had, so `app/api/routes.py` did not
change at its two job-submission sites.

---

## Deploying to CML

1. Put this folder in a CML project (git or upload). `static/` must come with
   it — CML runtimes have no Node to build the UI.
2. **New Session** → run:
   ```
   python bootstrap.py --demo 3000
   ```
   This installs `requirements.txt` and generates a demo dataset. The Session's
   environment persists with the project, so the Application inherits it.
3. **Applications → New Application**
   - Script: `cml_app.py`
   - Kernel: Python 3.10+
   - Resources: 2 vCPU / 8 GB is comfortable for the demo sizes
4. Open the Application URL. The UI and the API are both there —
   API docs at `/docs`, health at `/health`.

`.project-metadata.yaml` declares the same two steps, so the folder can also be
deployed as an AMP.

### Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `CDSW_APP_PORT` | set by CML | Port to bind. Falls back to 8100 off-platform. |
| `ATLAS_ENABLED` | `false` | Set true only if this workspace can reach Atlas. |
| `ATLAS_BASE_URL` | `http://localhost:21000` | Used only when Atlas is enabled. |
| `JOB_WORKERS` | `2` | Concurrent training/generation threads. |

---

## Running it locally, without Docker

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Scripts/ on Windows, bin/ on Linux
.venv/Scripts/python cml_app.py
```

Then open <http://localhost:8100>.

To edit the UI, run Vite separately (`npm --prefix frontend run dev`, proxying
`/api` to 8100) and rebuild with `python build_ui.py` when done.

---

## Verified on this build

Run end to end with no Docker, no Postgres, no Redis, no MinIO:

* upload → configure → autoconfigure → train → generate → preview → CSV download
* `scripts/smoke_test.py --rows 900 --beta 100` → **24/24 checks passed**
* Atlas policy resolved from the bundled fixture: 11 columns pseudonymised,
  1 generalised, `requires_dp: true` — DP switched on by the catalog, not by a
  user ticking a box
* Trained model: fidelity **0.903** at 800 customers, `dp_forced: true`

### Watch `beta`

Unchanged from the Docker build and worth repeating: the vendor default is
`100000`. If `beta` is at or above your row count every column becomes
independent, the model trains cleanly and reproduces **no correlations at all**.
Scale it to the table — roughly 100–200 for the demo sizes above.

---

## Differences that matter operationally

1. **Jobs die with the Application.** There is no broker holding a queue. On
   startup, jobs left `RUNNING` or `QUEUED` by a previous process are marked
   `FAILED` with "interrupted by an application restart" rather than being left
   to look live forever.
2. **One process, one SQLite file.** `cml_app.py` runs a single uvicorn worker
   deliberately — a second worker would mean two thread pools writing the same
   database.
3. **No horizontal scale.** Concurrency is `JOB_WORKERS` threads in one
   container. That is the ceiling; the Compose build could add worker containers.
4. **`data/` is project state.** SQLite, uploads, models and generated output all
   live there and are gitignored. Deleting it resets the platform.

The honest limitations in `../poc/README.md` — DP's cost on small tables, the
fidelity score being a heuristic, shallow time-series support, no auth, not
load-tested — all still apply here unchanged.
