#!/usr/bin/env bash
# Install reviewed UI files only; preserve the server's backend, database and vectors.
set -euo pipefail
revision="${1:?Usage: bash scripts/deploy_ui.sh COMMIT_SHA}"
case "$revision" in (*[!0-9a-f]*|'') echo "Use a full commit SHA" >&2; exit 1;; esac
[ "${#revision}" -eq 40 ] || { echo "Use a full commit SHA" >&2; exit 1; }
cd "$(git rev-parse --show-toplevel)"
git cat-file -e "$revision^{commit}"
current_index_sha="$(sha256sum muhawir/static/index.html | cut -d ' ' -f 1)"
target_index_sha="$(git show "$revision:muhawir/static/index.html" | sha256sum | cut -d ' ' -f 1)"
if [ "$current_index_sha" = "$target_index_sha" ]; then
  echo "The reviewed UI is already installed."
  exit 0
fi
if [ "$current_index_sha" != "064699008a1620d4c97bd137441f11a0c46fabebc2310d6f14bba420e17cbad2" ] && [ "$current_index_sha" != "9386e3ec5fadc3cdfcc8e537bc56d6fe6d47b15754f3e451fc6453d75c6d9d8f" ]; then
  echo "Stopped: the server UI differs from the reviewed live version. Preserve and reconcile it before updating." >&2
  exit 1
fi
# This static-only update needs the stylesheet route already running on Oracle.
curl --fail --silent --max-time 15 http://127.0.0.1:8000/book-answer.css >/dev/null
ui_stage="$(mktemp -d .muhawir-ui-stage.XXXXXX)"
ui_backup="${MUHAWIR_UI_BACKUP_ROOT:-${HOME}}/muhawir-ui-backup-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$ui_backup"
files=(muhawir/static/fonts/Amiri-Regular.ttf muhawir/static/fonts/Amiri-Bold.ttf muhawir/static/fonts/Amiri-OFL.txt muhawir/static/book-answer.css muhawir/static/index.html)
for path in "${files[@]}"; do
  mkdir -p "$ui_stage/new/$(dirname "$path")" "$ui_backup/$(dirname "$path")"
  git show "$revision:$path" > "$ui_stage/new/$path"
  if [ -e "$path" ]; then cp -p "$path" "$ui_backup/$path"; else touch "$ui_backup/$(basename "$path").absent"; fi
done
installed=0
cleanup() {
  result=$?
  if [ "$result" -ne 0 ] && [ "$installed" -eq 1 ]; then
    for path in "${files[@]}"; do
      if [ -e "$ui_backup/$path" ]; then cp -p "$ui_backup/$path" "$path";
      elif [ -e "$ui_backup/$(basename "$path").absent" ]; then rm -f "$path"; fi
    done
    echo "Update failed; restored the previous UI." >&2
  fi
  rm -rf "$ui_stage"
}
trap cleanup EXIT
installed=1
for path in "${files[@]}"; do
  mkdir -p "$(dirname "$path")"
  chmod 644 "$ui_stage/new/$path"
  mv "$ui_stage/new/$path" "$path"
done
# No service restart and no answer request: all five assets are served from disk.
served_sha="$(curl --fail --silent --max-time 15 http://127.0.0.1:8000/ | sha256sum | cut -d ' ' -f 1)"
[ "$served_sha" = "$target_index_sha" ] || { echo "The service is serving a different directory." >&2; exit 1; }
curl --fail --silent --max-time 15 http://127.0.0.1:8000/book-answer.css >/dev/null
echo "UI deployed and verified. Previous files: $ui_backup"
