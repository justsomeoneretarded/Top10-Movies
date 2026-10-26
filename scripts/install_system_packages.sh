#!/usr/bin/env bash
# Install apt packages without letting a stalled Ubuntu mirror hang the job.
# Each attempt has a hard time limit; the retry uses Ubuntu's main archive.
# If FFmpeg still is not installed, a static build is fetched from GitHub.
# Usage: bash scripts/install_system_packages.sh ffmpeg tesseract-ocr ...
set -uo pipefail

packages=("$@")
options=(-o Acquire::Retries=2 -o Acquire::http::Timeout=20 -o Acquire::https::Timeout=20)

attempt() {
  timeout 240 sudo apt-get "${options[@]}" update -qq &&
  timeout 300 sudo env DEBIAN_FRONTEND=noninteractive apt-get "${options[@]}" install -y --no-install-recommends "${packages[@]}"
}

if attempt; then
  exit 0
fi
echo "::warning title=Package mirror slow::Retrying with Ubuntu's main archive."
sudo sed -i 's#http://azure.archive.ubuntu.com#http://archive.ubuntu.com#g' /etc/apt/sources.list /etc/apt/sources.list.d/*.sources 2>/dev/null || true
if attempt; then
  exit 0
fi

# Last resort for the one package everything needs: a static FFmpeg from GitHub releases.
if ! command -v ffmpeg >/dev/null; then
  echo "::warning title=Package mirror unavailable::Installing a static FFmpeg build from GitHub instead."
  url="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-linux64-gpl-7.1.tar.xz"
  curl --fail --silent --show-error --location --retry 3 "$url" -o "$RUNNER_TEMP/ffmpeg.tar.xz" &&
    tar -xJf "$RUNNER_TEMP/ffmpeg.tar.xz" -C "$RUNNER_TEMP" &&
    sudo cp "$RUNNER_TEMP"/ffmpeg-*/bin/ffmpeg "$RUNNER_TEMP"/ffmpeg-*/bin/ffprobe /usr/local/bin/ || exit 1
fi
# Other packages are optional for the pipeline (OCR, fonts); report what is missing and continue.
for package in "${packages[@]}"; do
  dpkg -s "$package" >/dev/null 2>&1 || echo "::warning title=Optional package missing::$package could not be installed."
done
exit 0
