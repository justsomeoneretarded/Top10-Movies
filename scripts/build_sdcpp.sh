#!/usr/bin/env bash
# Build stable-diffusion.cpp (CPU, AVX2) into ./sdcpp and print the CLI path.
# The workflow caches ./sdcpp by upstream commit, so a rebuild only happens when it changes.
set -euo pipefail
commit="${1:-}"
if [ ! -d sdcpp/.git ]; then
  git clone -q --recursive https://github.com/leejet/stable-diffusion.cpp sdcpp
fi
cd sdcpp
if [ -n "$commit" ]; then git fetch -q --depth 1 origin "$commit" && git checkout -q "$commit" && git submodule update -q --init --recursive; fi
binary=""
for name in build/bin/sd-cli build/bin/sd; do [ -x "$name" ] && binary="$name" && break; done
if [ -z "$binary" ]; then
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON >/dev/null
  cmake --build build -j "$(nproc)" >/dev/null
  for name in build/bin/sd-cli build/bin/sd; do [ -x "$name" ] && binary="$name" && break; done
fi
[ -n "$binary" ] || { echo "::error title=stable-diffusion.cpp build::no CLI binary was produced"; exit 1; }
echo "SD_BINARY=$PWD/$binary" >> "${GITHUB_ENV:-/dev/null}"
echo "$PWD/$binary"
