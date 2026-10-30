#!/usr/bin/env bash
# Re-export every .drawio file in this directory to PNG.
# Requires the draw.io desktop CLI (`drawio` in PATH).
set -euo pipefail
cd "$(dirname "$0")"
for f in *.drawio; do
  drawio -x -f png -o "${f%.drawio}.png" --scale 2 --border 20 --crop "$f"
done
