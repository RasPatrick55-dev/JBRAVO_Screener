#!/usr/bin/env bash
set -euo pipefail

# build_frontend_pythonanywhere.sh
# - Loads nvm so node/npm are available in the PythonAnywhere shell.
# - Installs JS deps and builds the Vite frontend.
# - Ensures the build output is in the directory served by the WSGI app.
#
# Usage inside PythonAnywhere:
#   chmod +x build_frontend_pythonanywhere.sh
#   ./build_frontend_pythonanywhere.sh
#
# This enables you to build the React frontend directly on PythonAnywhere.
# Caveats:
# - npm ci requires a package-lock.json (present in this repo).
# - Ensure VITE_LOGO_DEV_API_KEY is set in frontend/.env for logos.

REPO_DIR="${REPO_DIR:-$HOME/jbravo_screener}"
FRONTEND_DIR="$REPO_DIR/frontend"
BUILD_DIR="$FRONTEND_DIR/dist"
TARGET_DIR="$FRONTEND_DIR/dist"
NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
NODE_VERSION="${NODE_VERSION:-20.19.0}"

log() {
  printf "[frontend-build] %s\n" "$1"
}

if [ ! -d "$FRONTEND_DIR" ]; then
  log "Frontend directory not found: $FRONTEND_DIR"
  exit 1
fi

if [ -s "$HOME/.bashrc" ]; then
  # shellcheck disable=SC1090
  . "$HOME/.bashrc"
fi
if [ -s "$HOME/.profile" ]; then
  # shellcheck disable=SC1090
  . "$HOME/.profile"
fi
if [ -s "$NVM_DIR/nvm.sh" ]; then
  # shellcheck disable=SC1090
  . "$NVM_DIR/nvm.sh"
fi

if command -v nvm >/dev/null 2>&1; then
  log "Using Node.js $NODE_VERSION via nvm"
  nvm install "$NODE_VERSION"
  nvm use "$NODE_VERSION"
fi

if ! command -v node >/dev/null 2>&1; then
  log "node not found. Run ./install_node_pythonanywhere.sh first."
  exit 1
fi

NODE_CURRENT="$(node -v | sed 's/^v//')"
NODE_MAJOR="${NODE_CURRENT%%.*}"
NODE_MINOR="${NODE_CURRENT#*.}"
NODE_MINOR="${NODE_MINOR%%.*}"
if [ "$NODE_MAJOR" -lt 20 ] || { [ "$NODE_MAJOR" -eq 20 ] && [ "$NODE_MINOR" -lt 19 ]; }; then
  log "Node.js $NODE_CURRENT is too old. Install $NODE_VERSION with ./install_node_pythonanywhere.sh."
  exit 1
fi

log "Building frontend in $FRONTEND_DIR"
cd "$FRONTEND_DIR"
npm ci
npm run build

if [ ! -d "$BUILD_DIR" ]; then
  log "Build output not found at $BUILD_DIR"
  exit 1
fi

# In this repo, the WSGI app serves frontend/dist directly.
if [ "$BUILD_DIR" = "$TARGET_DIR" ]; then
  log "Build output already in $TARGET_DIR; no copy needed."
else
  log "Syncing build output to $TARGET_DIR"
  rm -rf "$TARGET_DIR"
  mkdir -p "$TARGET_DIR"
  rsync -a --delete "$BUILD_DIR/" "$TARGET_DIR/"
fi

log "Build complete. Reload the web app to pick up new assets."
