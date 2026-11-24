#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/packaging/bin"
WORK_DIR="$(mktemp -d)"
PREFIX="$WORK_DIR/prefix"
JOBS="$(sysctl -n hw.ncpu)"

trap 'rm -rf "$WORK_DIR"' EXIT
mkdir -p "$OUT_DIR" "$PREFIX"

export HOMEBREW_NO_AUTO_UPDATE=1
export HOMEBREW_NO_INSTALL_CLEANUP=1
for tool in meson ninja pkg-config; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    brew install "$tool"
  fi
done

echo "=== [1/3] Building static libmp3lame ==="
cd "$WORK_DIR"
curl -fsSL https://downloads.sourceforge.net/project/lame/lame/3.100/lame-3.100.tar.gz -o lame.tar.gz
tar -xzf lame.tar.gz
cd lame-3.100
./configure --prefix="$PREFIX" --disable-shared --enable-static --disable-frontend --disable-decoder
make -j"$JOBS" install

echo "=== [2/3] Building static librubberband (Accelerate vDSP) ==="
cd "$WORK_DIR"
git clone --depth 1 --branch v4.0.0 https://github.com/breakfastquay/rubberband.git
cd rubberband
meson setup build \
  --prefix="$PREFIX" \
  --libdir=lib \
  --default-library=static \
  -Dcpp_args="-include stddef.h" \
  -Dfft=vdsp \
  -Dresampler=builtin \
  -Dcmdline=disabled \
  -Djni=disabled \
  -Dladspa=disabled \
  -Dlv2=disabled \
  -Dvamp=disabled \
  -Dtests=disabled
ninja -C build -j"$JOBS" install

echo "=== [3/3] Building static audio-only FFmpeg + FFprobe ==="
cd "$WORK_DIR"
git clone --depth 1 --branch n7.1.1 https://github.com/FFmpeg/FFmpeg.git
cd FFmpeg
PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig" ./configure \
  --prefix="$PREFIX" \
  --pkg-config-flags="--static" \
  --extra-cflags="-I$PREFIX/include" \
  --extra-ldflags="-L$PREFIX/lib -framework Accelerate -lc++" \
  --disable-shared \
  --enable-static \
  --disable-autodetect \
  --enable-zlib \
  --enable-audiotoolbox \
  --disable-doc \
  --disable-htmlpages \
  --disable-manpages \
  --disable-podpages \
  --disable-txtpages \
  --disable-ffplay \
  --disable-network \
  --disable-everything \
  --enable-gpl \
  --enable-libmp3lame \
  --enable-librubberband \
  --enable-ffmpeg \
  --enable-ffprobe \
  --enable-protocol=file,pipe \
  --enable-demuxer=aac,ac3,aiff,ape,asf,au,caf,flac,matroska,mov,mp3,ogg,pcm_f32le,pcm_s16le,pcm_s24le,pcm_s32le,pcm_u8,wav,wv \
  --enable-muxer=mp3,wav,pcm_f32le,pcm_s16le,null \
  --enable-decoder=aac,aac_fixed,aac_latm,ac3,alac,ape,eac3,flac,mp1,mp1float,mp2,mp2float,mp3,mp3adu,mp3adufloat,mp3float,mp3on4,mp3on4float,opus,pcm_f32be,pcm_f32le,pcm_f64be,pcm_f64le,pcm_s16be,pcm_s16le,pcm_s24be,pcm_s24le,pcm_s32be,pcm_s32le,pcm_u8,vorbis,wavpack,wmav1,wmav2,wmalossless,wmapro \
  --enable-encoder=libmp3lame,pcm_f32le,pcm_s16le \
  --enable-parser=aac,aac_latm,ac3,flac,mpegaudio,opus,vorbis \
  --enable-bsf=aac_adtstoasc \
  --enable-filter=abuffer,abuffersink,aformat,anull,aresample,asetpts,atempo,atrim,loudnorm,rubberband,volume
make -j"$JOBS" install

cp "$PREFIX/bin/ffmpeg" "$OUT_DIR/ffmpeg"
cp "$PREFIX/bin/ffprobe" "$OUT_DIR/ffprobe"
chmod +x "$OUT_DIR/ffmpeg" "$OUT_DIR/ffprobe"

echo "=== Verifying zero Homebrew dylib dependencies ==="
otool -L "$OUT_DIR/ffmpeg"
otool -L "$OUT_DIR/ffprobe"
if otool -L "$OUT_DIR/ffmpeg" "$OUT_DIR/ffprobe" | grep -q "/opt/homebrew"; then
  echo "ERROR: Unexpected /opt/homebrew dynamic dependency in static FFmpeg!" >&2
  exit 1
fi
"$OUT_DIR/ffmpeg" -filters | grep -q rubberband
"$OUT_DIR/ffmpeg" -encoders | grep -q libmp3lame
echo "Static macOS FFmpeg + FFprobe ready in $OUT_DIR"
