#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="${SRC_ROOT:-/home/s1/Desktop/datasets/pick_place_brown_300}"
DST_ROOT="${DST_ROOT:-/home/s1/Desktop/datasets/pick_place_brown_300_fastseek}"
JOBS="${JOBS:-16}"
CRF="${CRF:-22}"
GOP="${GOP:-2}"
V_CODEC="${V_CODEC:-libx264}"
PIX_FMT="${PIX_FMT:-yuv420p}"
PRESET="${PRESET:-fast}"

if [[ ! -d "${SRC_ROOT}" ]]; then
  echo "SRC_ROOT does not exist: ${SRC_ROOT}"
  exit 1
fi

mkdir -p "${DST_ROOT}"

echo "Copying metadata/data files..."
rsync -a --delete \
  --exclude 'videos' \
  "${SRC_ROOT}/" "${DST_ROOT}/"

echo "Re-encoding videos for fast random access..."
find "${SRC_ROOT}/videos" -type f -name '*.mp4' > /tmp/smolvla_reencode_list.txt

cat /tmp/smolvla_reencode_list.txt | xargs -I{} -P "${JOBS}" bash -lc '
  set -euo pipefail
  src="$1"
  src_root="$2"
  dst_root="$3"
  vcodec="$4"
  pix_fmt="$5"
  preset="$6"
  crf="$7"
  gop="$8"

  rel="${src#${src_root}/}"
  dst="${dst_root}/${rel}"
  mkdir -p "$(dirname "${dst}")"

  ffmpeg -hide_banner -loglevel error -y \
    -i "${src}" \
    -an \
    -c:v "${vcodec}" \
    -pix_fmt "${pix_fmt}" \
    -preset "${preset}" \
    -crf "${crf}" \
    -g "${gop}" -keyint_min "${gop}" -sc_threshold 0 \
    "${dst}"
' _ {} "${SRC_ROOT}" "${DST_ROOT}" "${V_CODEC}" "${PIX_FMT}" "${PRESET}" "${CRF}" "${GOP}"

echo "Done."
echo "New dataset: ${DST_ROOT}"
