#!/usr/bin/env bash
set -e

echo "[1/3] Updating apt package list..."
sudo apt update

echo "[2/3] Installing build dependencies..."
sudo apt install -y \
  build-essential \
  gcc \
  g++ \
  make \
  python3-dev
  cmake

echo "[3/3] Checking installed tools..."
echo -n "gcc: "
gcc --version | head -n 1

echo -n "g++: "
g++ --version | head -n 1

echo -n "make: "
make --version | head -n 1

echo "Done. Build dependencies installed."