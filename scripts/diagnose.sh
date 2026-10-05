#!/usr/bin/env bash
# One report of the server's state, safe to share: every key, token or password is hidden.
# Usage (on the server, in ~/muhawir):  scripts/diagnose.sh > ~/diagnose.txt
set -u
cd "$(dirname "$0")/.."
hide='s/((KEY|TOKEN|SECRET|PASSWORD)[A-Z_]*=).*/\1(hidden)/'

echo "== time";            date -u
echo "== code";            git log --oneline -1; git status --short | head -5
echo "== settings (keys hidden)"
sed -E "$hide" ~/muhawir.env | grep -v '^\s*$'
echo "== service";         systemctl status muhawir --no-pager 2>&1 | head -6
echo "== unit file";       systemctl cat muhawir --no-pager 2>&1 | grep -E 'EnvironmentFile|ExecStart|Restart'
echo "== timers and other muhawir units"
systemctl list-timers --all --no-pager 2>&1 | grep -i muhawir
for u in $(systemctl list-unit-files --no-pager 2>/dev/null | awk '/muhawir/ {print $1}'); do
  [ "$u" = "muhawir.service" ] && continue
  echo "-- $u"; systemctl cat "$u" --no-pager 2>&1 | grep -E 'Exec|OnCalendar|OnUnitActive|OnBoot'
done
echo "== restarts today";  journalctl -u muhawir --since today --no-pager 2>/dev/null | grep -c 'Started muhawir'
echo "== caddy";           systemctl is-active caddy; grep -v '^\s*#' /etc/caddy/Caddyfile 2>/dev/null | grep -v '^\s*$'
echo "== https check"
for host in muhawir.duckdns.org muhawir.84.13.157.247.sslip.io; do
  printf '%s ' "$host"; curl -s -o /dev/null -w '%{http_code} %{time_total}s\n' --max-time 15 "https://$host/" || echo failed
done
echo "== local health";    curl -s --max-time 10 http://127.0.0.1:8000/api/health || echo "no answer"
echo
echo "== memory and disk";  free -m | head -2; df -h / | tail -1
echo "== data";             ls -la data/ 2>/dev/null | grep -E 'muhawir.db|vectors'
echo "== model errors (last 2 h)"
journalctl -u muhawir --since "2 hours ago" --no-pager 2>/dev/null | grep -iE 'failed|error|429|timeout' | sed -E "$hide" | cut -c1-200 | tail -15
echo "== model calls (last 30 min)"
journalctl -u muhawir --since "30 min ago" --no-pager 2>/dev/null | grep ' timing ' | cut -c40-160 | tail -25
echo "== not answered (last 2 h)"
journalctl -u muhawir --since "2 hours ago" --no-pager 2>/dev/null | grep -E 'not answered' | cut -c1-200 | tail -10
