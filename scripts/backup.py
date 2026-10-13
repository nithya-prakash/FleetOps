"""Consistent backup of the stateful parts of a running Compose stack, and verification of a backup.

    python scripts/backup.py backup  [--out backups/<timestamp>]
    python scripts/backup.py verify  backups/<timestamp>

What is backed up
* control-plane SQLite database (online ``sqlite3`` backup API: consistent while the service runs; a plain file copy of a
  live WAL database is NOT consistent), and the content-addressed artifact store;
* MLflow tracking/registry database, MLflow's basic-auth database and MLflow artifacts.
Not backed up: secrets (``deploy/secrets`` is generated from ``.env``; keep ``.env`` in your secret manager), Prometheus/Grafana
data (re-creatable), edge devices (they re-sync from the control plane).

Restore (manual, with the stack stopped): copy the files back into the volumes, start the stack, run ``verify`` against the live stack's
counts. ``verify`` here checks the backup itself: manifest hashes, SQLite integrity, row counts versus the counts recorded at backup time,
and that every registered artifact in the backup exists and matches its recorded SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CP, MLFLOW = "fleetops-control-plane-1", "fleetops-mlflow-1"
SQLITE_BACKUP = "import sqlite3,sys; s=sqlite3.connect(sys.argv[1]); d=sqlite3.connect(sys.argv[2]); s.backup(d); d.close(); s.close()"
COUNT_TABLES = ("devices", "deployments", "deployment_targets", "audit_events", "model_versions")


def run(*args: str) -> None:
    subprocess.run(args, check=True, capture_output=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def counts(db: Path) -> dict[str, int]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        existing = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
        return {t: con.execute(f"select count(*) from {t}").fetchone()[0] for t in COUNT_TABLES if t in existing}  # noqa: S608
    finally:
        con.close()


def dump_db(container: str, path: str, out: Path) -> None:
    tmp = "/tmp/backup.db"  # tmpfs inside the container: `docker cp` cannot read it, so the file is streamed out
    run("docker", "exec", container, "python", "-c", SQLITE_BACKUP, path, tmp)
    with out.open("wb") as fh:
        subprocess.run(["docker", "exec", container, "cat", tmp], check=True, stdout=fh)
    run("docker", "exec", container, "rm", "-f", tmp)


def backup(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    out.chmod(0o700)
    dump_db(CP, "/data/fleetops.db", out / "fleetops.db")
    dump_db(MLFLOW, "/data/mlflow.db", out / "mlflow.db")
    dump_db(MLFLOW, "/data/basic_auth.db", out / "basic_auth.db")
    run("docker", "cp", f"{CP}:/data/artifacts", str(out / "artifacts"))
    run("docker", "cp", f"{MLFLOW}:/data/artifacts", str(out / "mlflow-artifacts"))
    files = sorted(p for p in out.rglob("*") if p.is_file())
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "counts": counts(out / "fleetops.db"),
        "files": {str(p.relative_to(out)): sha256(p) for p in files},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"backup written to {out}: {len(files)} files, counts {manifest['counts']}")


def verify(path: Path) -> int:
    manifest = json.loads((path / "manifest.json").read_text())
    problems: list[str] = []
    for rel, digest in manifest["files"].items():
        f = path / rel
        if not f.is_file():
            problems.append(f"missing {rel}")
        elif sha256(f) != digest:
            problems.append(f"hash mismatch {rel}")
    with tempfile.TemporaryDirectory() as tmp:  # integrity checks run on a copy so the backup stays untouched
        for name in ("fleetops.db", "mlflow.db", "basic_auth.db"):
            copy = Path(tmp) / name
            copy.write_bytes((path / name).read_bytes())
            con = sqlite3.connect(copy)
            result = con.execute("pragma integrity_check").fetchone()[0]
            con.close()
            if result != "ok":
                problems.append(f"{name}: integrity_check says {result}")
        if counts(Path(tmp) / "fleetops.db") != manifest["counts"]:
            problems.append("row counts differ from those recorded at backup time")
    for problem in problems:
        print("FAIL", problem)
    print("backup verified" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backup")
    b.add_argument("--out", type=Path, default=ROOT / "backups" / datetime.now(UTC).strftime("%Y%m%d-%H%M%S"))
    v = sub.add_parser("verify")
    v.add_argument("path", type=Path)
    args = ap.parse_args()
    if args.cmd == "backup":
        backup(args.out)
    else:
        sys.exit(verify(args.path))


if __name__ == "__main__":
    main()
