#!/bin/bash
# The pinned reSIDfp to WebAssembly, from the patched sources the linux wheel was built from.
#
#   ~/s06_wasm/build.sh
#
# The tree is ~/pyresidfp-repro/build-linux/pyresidfp-0.17.0, which is the 0.17.0 sdist with the
# two reproducibility patches applied: the dither ring removed from FilterModelConfig.h and the
# resampler ring zero-initialised in SincResampler.h. config.h and siddefs-fp.h are generated
# here with the settings CMakeLists.txt configures them with: C++20, no SMMINTRIN, no ARM NEON,
# inlining on, branch hints on.
set -euo pipefail
ROOT=~/s06_wasm
SRC=~/s06_wasm/src   # the pinned patched tree, copied, with the table builds off the thread pool
export EMSDK_PYTHON=/usr/bin/python3.11
mkdir -p "$ROOT/gen" "$ROOT/out" "$ROOT/bin"
ln -sf /usr/bin/python3.11 "$ROOT/bin/python3"       # emcc runs the python on PATH, and the system one is 3.9
export PATH="$ROOT/bin:$PATH"
source ~/emsdk/emsdk_env.sh > /dev/null 2>&1

cat > "$ROOT/gen/config.h" <<'H'
#define HAVE_CXX11
#define HAVE_CXX14
#define HAVE_CXX17
#define HAVE_CXX20
#define HAVE_CXX23
/* #undef HAVE_ARM_NEON_H */
/* #undef HAVE_SMMINTRIN_H */
H

sed -e 's/#cmakedefine01 RESID_BRANCH_HINTS/#define RESID_BRANCH_HINTS 1/' \
    -e 's/#cmakedefine01 HAVE_BUILTIN_EXPECT/#define HAVE_BUILTIN_EXPECT 1/' \
    -e 's/#cmakedefine01 RESID_INLINING/#define RESID_INLINING 1/' \
    -e 's/#cmakedefine RESID_INLINE @RESID_INLINE@/#define RESID_INLINE inline/' \
    -e 's/@PACKAGE_VERSION@/0.17.0/' \
    "$SRC/src/residfp/siddefs-fp.h.in" > "$ROOT/gen/siddefs-fp.h"

em++ --version | head -1
em++ -O3 -DNDEBUG -DHAVE_CONFIG_H -DPROJECT_VERSION='"0.17.0"' -std=c++20 -flto -fexceptions \
  -I "$ROOT/gen" -I "$SRC/src" -I "$SRC/src/residfp" -I "$SRC/src/residfp/resample" \
  "$ROOT/shim.cpp" \
  "$SRC/src/PythonSid.cpp" \
  "$SRC/src/residfp/resample/SincResampler.cpp" \
  "$SRC/src/residfp/Dac.cpp" \
  "$SRC/src/residfp/EnvelopeGenerator.cpp" \
  "$SRC/src/residfp/ExternalFilter.cpp" \
  "$SRC/src/residfp/Filter.cpp" \
  "$SRC/src/residfp/Filter6581.cpp" \
  "$SRC/src/residfp/Filter8580.cpp" \
  "$SRC/src/residfp/FilterModelConfig.cpp" \
  "$SRC/src/residfp/FilterModelConfig6581.cpp" \
  "$SRC/src/residfp/FilterModelConfig8580.cpp" \
  "$SRC/src/residfp/Integrator6581.cpp" \
  "$SRC/src/residfp/Integrator8580.cpp" \
  "$SRC/src/residfp/OpAmp.cpp" \
  "$SRC/src/residfp/SID.cpp" \
  "$SRC/src/residfp/Spline.cpp" \
  "$SRC/src/residfp/version.cc" \
  "$SRC/src/residfp/WaveformCalculator.cpp" \
  "$SRC/src/residfp/WaveformGenerator.cpp" \
  -o "$ROOT/out/sid.mjs" \
  -s MODULARIZE=1 -s EXPORT_ES6=1 -s ENVIRONMENT=web,node -s ALLOW_MEMORY_GROWTH=1 \
  -s INITIAL_MEMORY=33554432 \
  -s EXPORTED_FUNCTIONS='["_sid_create","_sid_destroy","_sid_reset","_sid_enable_filter","_sid_write","_sid_read","_sid_mute","_sid_input","_sid_clock","_sid_last_error","_malloc","_free"]' \
  -s EXPORTED_RUNTIME_METHODS='["cwrap","HEAP16","HEAPU8","UTF8ToString"]'

ls -l "$ROOT/out"
sha256sum "$ROOT/out/sid.wasm" "$ROOT/out/sid.mjs"
