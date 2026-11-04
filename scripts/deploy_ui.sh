#!/usr/bin/env bash
# Retired (audit M-012, November 2026). This copied a fixed list of five UI files from one 2026 release and reported
# success when index.html alone matched, so newer stylesheets and assets could be skipped silently.
# The whole application, interface included, is updated together:
#   cd ~/muhawir && git pull --ff-only origin main && sudo systemctl restart muhawir
echo "scripts/deploy_ui.sh is retired: update with  git pull --ff-only origin main  then restart the service." >&2
exit 1
