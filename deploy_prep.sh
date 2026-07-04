#!/usr/bin/env bash
# Prepare a clean upload folder for ModelScope Studio drag-and-drop deployment.
# Usage: ./deploy_prep.sh
# Then drag /tmp/constructiv-upload/ into ModelScope Studio.

set -euo pipefail

DEST="/tmp/constructiv-upload"

echo "→ Clearing previous upload..."
rm -rf "$DEST"
mkdir -p "$DEST"

echo "→ Syncing files..."
rsync -a \
  --exclude='.env' \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='.claude/' \
  --exclude='data/' \
  --exclude='tests/' \
  --exclude='icap_harness.py' \
  --exclude='SESSION_SUMMARY.md' \
  --exclude='.git/' \
  --exclude='Dockerfile' \
  --exclude='.dockerignore' \
  --exclude='deploy_prep.sh' \
  /home/el/Desktop/qwen/ "$DEST/"

echo ""
echo "✓ Ready at: $DEST"
echo ""
echo "Files included:"
find "$DEST" -type f | sed "s|$DEST/||" | sort
echo ""
echo "Next: drag $DEST into ModelScope Studio → Redeploy"
