#!/bin/sh
# Downloads the bge-small embedding model (bge_small_buckets_float16.aimodel, 64 MB) from this
# repository's GitHub release into the given folder (default: the current folder).
#   Tools/download_model.sh [destination-folder] [release-tag]
# Uses the GitHub CLI if installed, otherwise curl (the repository is public; no login needed).
set -eu
REPO="DavidOrtegit123/OnDeviceRag"
DEST="${1:-.}"
TAG="${2:-0.2.0}"
ASSET="bge_small_buckets_float16.aimodel.zip"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if command -v gh >/dev/null 2>&1; then
    gh release download "$TAG" --repo "$REPO" --pattern "$ASSET" --dir "$TMP"
else
    curl -fL -o "$TMP/$ASSET" "https://github.com/$REPO/releases/download/$TAG/$ASSET"
fi
mkdir -p "$DEST"
rm -rf "$DEST/bge_small_buckets_float16.aimodel"
unzip -q "$TMP/$ASSET" -d "$DEST"
echo "Model ready: $DEST/bge_small_buckets_float16.aimodel"
