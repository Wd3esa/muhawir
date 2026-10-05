#!/usr/bin/env bash
# Fetch the vector index built on Kaggle, check it against the server's database, and switch to it.
#
# Usage (on the server, in ~/muhawir):
#   scripts/fetch_vectors.sh <kaggle-username>/<notebook-slug>
#
# Needs a Kaggle API token in ~/.kaggle/kaggle.json (kaggle.com → Settings → API → Create New Token),
# readable by this user only (chmod 600). The token can only read the Kaggle account: the server's own
# keys never leave the server. See docs/OPERATIONS.md.
#
# The new index replaces the old one only if its passage ids are exactly the database's, in order;
# otherwise nothing changes and the script says why. The old index is kept as data/vectors.previous/.
set -euo pipefail

notebook="${1:-}"
if [[ -z "$notebook" || "$notebook" != */* ]]; then
  echo "usage: $0 <kaggle-username>/<notebook-slug>" >&2
  exit 2
fi
cd "$(dirname "$0")/.."
[[ -f ~/.kaggle/kaggle.json ]] || { echo "missing ~/.kaggle/kaggle.json (Kaggle API token)" >&2; exit 2; }
chmod 600 ~/.kaggle/kaggle.json

.venv/bin/pip install -q kaggle
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

echo "downloading the output of $notebook ..."
.venv/bin/kaggle kernels output "$notebook" -p "$work" >/dev/null
archive="$(find "$work" -name 'muhawir-vectors.tar.gz' | head -1)"
[[ -n "$archive" ]] || { echo "no muhawir-vectors.tar.gz in the notebook's output" >&2; exit 1; }
mkdir "$work/new"
tar xzf "$archive" -C "$work/new"

echo "checking it against data/muhawir.db ..."
.venv/bin/python - "$work/new" <<'PY'
import json, sqlite3, sys
from pathlib import Path
new = Path(sys.argv[1])
ids = json.loads((new / "vectors_ids.json").read_text(encoding="utf-8"))
db = [r[0] for r in sqlite3.connect("data/muhawir.db").execute("SELECT id FROM passages ORDER BY rid")]
if ids != db:
    sys.exit(f"the index does not match the database ({len(ids)} vectors, {len(db)} passages): "
             "rebuild the database or run the notebook again with the same sources")
print(f"match: {len(ids)} passages")
PY

mkdir -p data/vectors.previous
for f in vectors.npy vectors_ids.json vectors_model.txt; do
  [[ -f "data/$f" ]] && mv "data/$f" "data/vectors.previous/$f"
  mv "$work/new/$f" "data/$f"
done
rm -f data/vectors.partial*
sudo systemctl restart muhawir
echo "done: the new index is in use (the old one is in data/vectors.previous/)"
