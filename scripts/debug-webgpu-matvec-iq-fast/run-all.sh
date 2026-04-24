#!/usr/bin/env bash
# One-command repro: builds master + branch, benches both, renders charts + summary.
# Leaves you on the branch you started on.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$REPO_ROOT"

MASTER_REF="${MASTER_REF:-master}"
BRANCH_REF="${BRANCH_REF:-webgpu-matvec-iq-fast}"
BUILD_DIR="${BUILD_DIR:-build-webgpu-profile}"
export BUILD_DIR

start_branch="$(git rev-parse --abbrev-ref HEAD)"
echo "==> starting on $start_branch; will restore at the end"

# Refuse to run with dirty tree — switching branches would clobber work.
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "error: working tree has uncommitted changes. Commit or stash first." >&2
  exit 1
fi

echo "==> step 1/5: fetch models"
bash "$HERE/fetch-models.sh"

for ref in "$MASTER_REF" "$BRANCH_REF"; do
  if [ "$ref" = "$MASTER_REF" ]; then tag="master"; else tag="branch"; fi
  echo
  echo "==> step: checkout $ref (tag=$tag)"
  git checkout "$ref"

  echo "==> step: build"
  bash "$HERE/build.sh"

  echo "==> step: bench models"
  bash "$HERE/bench-models.sh" "$tag"

  echo "==> step: bench kernel"
  bash "$HERE/bench-kernel.sh" "$tag"
done

echo
echo "==> step 5/5: render charts + summary"
VENV="/tmp/plot-venv-$USER"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
  "$VENV/bin/pip" install --quiet matplotlib numpy
fi
"$VENV/bin/python" "$HERE/plot.py"

echo
echo "==> restoring $start_branch"
git checkout "$start_branch"

echo
echo "==> DONE. Results in: $HERE/results/"
echo "    summary:   $HERE/results/summary.md"
echo "    charts:    chart-kernel.png  chart-decode.png  chart-prompt.png"
