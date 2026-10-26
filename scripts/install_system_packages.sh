#!/usr/bin/env bash
# Install apt packages without letting a stalled or broken package list fail or hang the job.
# - `apt-get update` errors (often from an unrelated third-party list) are reported, not fatal.
# - Each step has a hard time limit; a retry switches to Ubuntu's main archive.
# - If FFmpeg is still missing, a static build is installed from GitHub releases.
# Usage: bash scripts/install_system_packages.sh ffmpeg tesseract-ocr ...
set -uo pipefail

packages=("$@")
options=(-o Acquire::Retries=2 -o Acquire::http::Timeout=20 -o Acquire::https::Timeout=20)
log="${RUNNER_TEMP:-/tmp}/apt-install.log"

report() {
  local title="$1"
  local detail
  detail="$(tail -n 6 "$log" 2>/dev/null | tr '\n' ' ' | cut -c1-900)"
  echo "::warning title=$title::$detail"
}

attempt() {
  : > "$log"
  if ! timeout 240 sudo apt-get "${options[@]}" update -qq >>"$log" 2>&1; then
    report "apt-get update reported a problem (continuing)"
  fi
  timeout 420 sudo env DEBIAN_FRONTEND=noninteractive apt-get "${options[@]}" install -y --no-install-recommends \
    "${packages[@]}" >>"$log" 2>&1
}

if attempt; then
  exit 0
fi
report "Package install failed on the default mirror; retrying with Ubuntu's main archive"
sudo sed -i 's#http://azure.archive.ubuntu.com#http://archive.ubuntu.com#g' /etc/apt/sources.list /etc/apt/sources.list.d/*.sources 2>/dev/null || true
if attempt; then
  exit 0
fi
report "Package install failed again"

# Last resort for the one package everything needs: a static FFmpeg from GitHub releases.
if ! command -v ffmpeg >/dev/null || ! command -v ffprobe >/dev/null; then
  echo "::warning title=Installing static FFmpeg::Using a static FFmpeg build from GitHub releases."
  url="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz"
  dir="${RUNNER_TEMP:-/tmp}/ffmpeg-static"
  mkdir -p "$dir"
  if curl --fail --silent --show-error --location --retry 3 "$url" -o "$dir/ffmpeg.tar.xz" &&
     tar -xJf "$dir/ffmpeg.tar.xz" -C "$dir"; then
    sudo cp "$dir"/ffmpeg-*/bin/ffmpeg "$dir"/ffmpeg-*/bin/ffprobe /usr/local/bin/
  else
    echo "::error title=FFmpeg unavailable::Neither apt nor the static build could install FFmpeg."
    exit 1
  fi
fi
# Other packages are optional for the pipeline (OCR, fonts); report what is missing and continue.
for package in "${packages[@]}"; do
  dpkg -s "$package" >/dev/null 2>&1 || echo "::warning title=Optional package missing::$package could not be installed."
done
exit 0
