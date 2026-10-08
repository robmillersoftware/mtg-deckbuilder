#!/usr/bin/env bash
# Point the sim worker at Forge's newest daily snapshot (new sets get card scripts
# there first), then rebuild and restart only the sim worker.
set -euo pipefail
cd "$(dirname "$0")/.."

url=$(curl -fsSL https://api.github.com/repos/Card-Forge/forge/releases/tags/daily-snapshots |
  python3 -c 'import json,sys; print(next(a["browser_download_url"] for a in json.load(sys.stdin)["assets"] if a["name"].endswith(".tar.bz2")))')
echo "Forge snapshot: $url"

touch .env
[ -z "$(tail -c1 .env)" ] || echo >> .env  # .env may lack a trailing newline
if grep -q '^FORGE_SNAPSHOT_URL=' .env; then
  sed -i.bak "s#^FORGE_SNAPSHOT_URL=.*#FORGE_SNAPSHOT_URL=$url#" .env && rm -f .env.bak
else
  echo "FORGE_SNAPSHOT_URL=$url" >> .env
fi

docker compose build sim-worker
docker compose up -d --no-deps sim-worker
