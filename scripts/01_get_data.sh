#!/usr/bin/env bash
# Download the OpenBiomechanics baseball pitching data used by PitchLab.
#
# Source: https://github.com/drivelineresearch/openbiomechanics (dataset-v1)
# License: CC BY-NC-SA 4.0. The data is NOT redistributed in this repo.
#
# Usage:  bash scripts/01_get_data.sh [target_dir]      (default: data/obp)
set -euo pipefail

DEST="${1:-data/obp}"
REPO="https://raw.githubusercontent.com/drivelineresearch/openbiomechanics/main"
REL="https://github.com/drivelineresearch/openbiomechanics/releases/download/dataset-v1"

mkdir -p "$DEST"
cd "$DEST"

echo "-> tables from the OBP repo"
curl -fsSL -o poi_metrics.csv      "$REPO/baseball_pitching/data/poi/poi_metrics.csv"
curl -fsSL -o metadata.csv         "$REPO/baseball_pitching/data/metadata.csv"
curl -fsSL -o data_dictionary.csv  "$REPO/baseball_pitching/data/data_dictionary.csv"
curl -fsSL -o LICENSE-DATA.md      "$REPO/LICENSE-DATA.md"

echo "-> release assets (about 70 MB)"
for f in pitching_landmarks.zip pitching_joint_angles.zip; do
  [ -f "$f" ] || curl -fL --retry 3 -o "$f" "$REL/$f"
done

echo "-> verifying SHA-256 against Driveline's published sums"
cat > checksums.sha256 <<'SUMS'
efcc078d5d8e77ed1fbf0b9f6a78f750c57a624165f71f30e34142b74db7ec22  pitching_landmarks.zip
b4c4854ffcc52a0f757e0b3a3c86acdc4a2fc2e50516a76e62aabdb916afed46  pitching_joint_angles.zip
SUMS
if command -v sha256sum >/dev/null; then sha256sum -c checksums.sha256
else shasum -a 256 -c checksums.sha256; fi

echo "-> unzipping (faster loads than reading the zips each time)"
mkdir -p lm ja
[ -f lm/landmarks.csv ]    || unzip -oq pitching_landmarks.zip    -d lm
[ -f ja/joint_angles.csv ] || unzip -oq pitching_joint_angles.zip -d ja

echo "done: $(pwd)"
