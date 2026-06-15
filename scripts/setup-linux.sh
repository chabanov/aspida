#!/usr/bin/env bash
# setup-linux.sh — provision a bare Ubuntu/Debian box (e.g. a DigitalOcean
# droplet) to build Aspida. Installs Alire + the GNAT/gprbuild toolchain into
# ~/.local/share/alire/toolchains, exactly where the Makefile looks for them.
#
# After this, `make server ARCH=native` builds the encrypted server on Linux
# (the Makefile auto-detects OS via `uname` and passes -XOS=linux to gprbuild;
# no macOS SDK flags are used — see shared.gpr).
#
# Usage:  bash scripts/setup-linux.sh && make server
set -euo pipefail

ALR_VERSION="${ALR_VERSION:-2.0.2}"
ARCH="$(uname -m)"   # x86_64 / aarch64

echo ">> apt deps"
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
  build-essential ca-certificates curl unzip git pkg-config

if ! command -v alr >/dev/null 2>&1; then
  echo ">> installing Alire $ALR_VERSION ($ARCH)"
  case "$ARCH" in
    x86_64)  ALR_ZIP="alr-${ALR_VERSION}-bin-x86_64-linux.zip" ;;
    aarch64) ALR_ZIP="alr-${ALR_VERSION}-bin-aarch64-linux.zip" ;;
    *) echo "unsupported arch: $ARCH" >&2; exit 1 ;;
  esac
  tmp="$(mktemp -d)"
  curl -fsSL -o "$tmp/alr.zip" \
    "https://github.com/alire-project/alire/releases/download/v${ALR_VERSION}/${ALR_ZIP}"
  unzip -q "$tmp/alr.zip" -d "$tmp"
  sudo install -m755 "$tmp/bin/alr" /usr/local/bin/alr
  rm -rf "$tmp"
fi

echo ">> selecting GNAT + gprbuild toolchain (lands in ~/.local/share/alire/toolchains)"
alr toolchain --select gnat_native gprbuild

echo ">> done. build with:  make server ARCH=native"
echo "   (GPU dev boxes: also install the CUDA/ROCm toolkit separately — Stream B)"
