#!/usr/bin/env bash
set -euo pipefail

# install_node_pythonanywhere.sh
# - Installs nvm (Node Version Manager) in the user's home directory.
# - Ensures nvm loads on new shells (bash login/interactive).
# - Installs a stable Node.js version (v20.19.0) and sets it as default.
# - Verifies node + npm are available.
#
# Usage inside PythonAnywhere:
#   chmod +x install_node_pythonanywhere.sh
#   ./install_node_pythonanywhere.sh
#
# This enables building React/Vite frontends directly on PythonAnywhere.
# Caveats:
# - PythonAnywhere free plans may have limited disk/CPU; builds can be slow.
# - Always keep Node pinned to a known version for reproducibility.

NODE_VERSION="${NODE_VERSION:-20.19.0}"
NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
NVM_INSTALL_URL="https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh"

log() {
  printf "[node-setup] %s\n" "$1"
}

download_nvm() {
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$NVM_INSTALL_URL"
    return 0
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -qO- "$NVM_INSTALL_URL"
    return 0
  fi
  return 1
}

ensure_profile_snippet() {
  local profile_file="$1"
  local marker_begin="# >>> jbravo nvm >>>"
  local marker_end="# <<< jbravo nvm <<<"
  if [ ! -f "$profile_file" ]; then
    touch "$profile_file"
  fi
  if ! grep -q "$marker_begin" "$profile_file"; then
    {
      echo ""
      echo "$marker_begin"
      echo "export NVM_DIR=\"\$HOME/.nvm\""
      echo "[ -s \"\$NVM_DIR/nvm.sh\" ] && . \"\$NVM_DIR/nvm.sh\""
      echo "$marker_end"
    } >>"$profile_file"
  fi
}

log "Installing nvm to $NVM_DIR"
download_nvm | bash

log "Configuring shell profiles to load nvm"
ensure_profile_snippet "$HOME/.bashrc"
ensure_profile_snippet "$HOME/.profile"

log "Loading nvm into current shell"
export NVM_DIR="$NVM_DIR"
# shellcheck disable=SC1090
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

log "Installing Node.js $NODE_VERSION via nvm"
nvm install "$NODE_VERSION"
nvm alias default "$NODE_VERSION"

log "Verifying node and npm"
node -v
npm -v

log "Done. Open a new bash console or run: source ~/.bashrc"
