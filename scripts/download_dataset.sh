#!/bin/bash

set -e

REPO_URL="https://github.com/hipe-eval/HIPE-OCRepair-2026-data.git"
TARGET_DIR="HIPE-OCRepair-2026-data"

if [ -d "$TARGET_DIR" ]; then
    echo "[INFO] Directory '$TARGET_DIR' already exists."
    cd "$TARGET_DIR"
    git pull
    cd ..
    echo "[SUCCESS] Repository successfully updated."
else
    echo "[INFO] Directory '$TARGET_DIR' not found."
    git clone "$REPO_URL"
    echo "[SUCCESS] Repository successfully cloned."
fi

