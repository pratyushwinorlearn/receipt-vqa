#!/usr/bin/env bash
# Convert a screen recording into a small, good-looking GIF for the README / LinkedIn.
# usage: scripts/make_demo_gif.sh input.mp4 assets/demo.gif [fps] [width]
set -euo pipefail
IN="${1:?input video}"
OUT="${2:-assets/demo.gif}"
FPS="${3:-12}"
WIDTH="${4:-900}"
PALETTE="$(mktemp --suffix=.png)"
ffmpeg -y -i "$IN" -vf "fps=$FPS,scale=$WIDTH:-1:flags=lanczos,palettegen" "$PALETTE"
ffmpeg -y -i "$IN" -i "$PALETTE" -lavfi "fps=$FPS,scale=$WIDTH:-1:flags=lanczos [x]; [x][1:v] paletteuse" "$OUT"
rm -f "$PALETTE"
echo "wrote $OUT"
