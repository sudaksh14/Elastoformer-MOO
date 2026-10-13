#!/usr/bin/env bash
# Lightweight checks that need no GPU, cluster or dataset: syntax, shell syntax, unit tests.
#
#   ./run_local_checks.sh            # uses `python` from PATH (set PYTHON=... to override)
#   ./run_local_checks.sh -k nested  # extra arguments go to pytest
#
# A passing run shows the code parses and the CPU unit tests pass. It says nothing about GPU
# behaviour or experiment results.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY="${PYTHON:-python}"

echo "== python syntax (no files written)"
"$PY" - <<'PY'
import ast, pathlib, sys
bad = []
for root in ("src", "tests", "scripts"):
    for path in pathlib.Path(root).rglob("*.py"):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            bad.append(f"{path}: {exc}")
if bad:
    sys.exit("syntax errors:\n" + "\n".join(bad))
print("ok")
PY

echo "== shell syntax"
for f in *.sh scripts/*.sh submit_*.slurm; do
  bash -n "$f"
  echo "ok $f"
done

echo "== unit tests"
"$PY" -m pytest -p no:cacheprovider "$@"
