#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDK_ROOT="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-}}"
DEFAULT_NDK="${SDK_ROOT:+${SDK_ROOT}/ndk/26.1.10909125}"
NDK="${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-${DEFAULT_NDK}}}"
API="${ANDROID_API:-24}"
NCNN_REPOSITORY="${NCNN_REPOSITORY:-https://github.com/Tencent/ncnn.git}"
NCNN_REF="${NCNN_REF:-d0d50631179e380e587fb188338f5683d2c93275}"
NCNN_SOURCE="${NCNN_SOURCE_DIR:-${ROOT}/pipeline_out/ncnn_source}"
NCNN_BUILD="${NCNN_BUILD_DIR:-${NCNN_SOURCE}/build-android-arm64}"
TOOLCHAIN="${NDK}/toolchains/llvm/prebuilt/linux-x86_64"
CLANG="${TOOLCHAIN}/bin/aarch64-linux-android${API}-clang"
AR="${TOOLCHAIN}/bin/llvm-ar"
STRIP="${TOOLCHAIN}/bin/llvm-strip"
CXX_SHARED="${TOOLCHAIN}/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
OUTPUT_DIR="${ROOT}/android/proprionav/src/main/jniLibs/arm64-v8a"

if [[ -z "${NDK}" || ! -x "${CLANG}" ]]; then
    echo "找不到 Android NDK clang: ${CLANG}" >&2
    echo "请设置 ANDROID_NDK_HOME 或 ANDROID_HOME" >&2
    exit 1
fi

if [[ ! -f "${NCNN_SOURCE}/CMakeLists.txt" ]]; then
    mkdir -p "${NCNN_SOURCE}"
    git -C "${NCNN_SOURCE}" init
    if git -C "${NCNN_SOURCE}" remote get-url origin >/dev/null 2>&1; then
        git -C "${NCNN_SOURCE}" remote set-url origin "${NCNN_REPOSITORY}"
    else
        git -C "${NCNN_SOURCE}" remote add origin "${NCNN_REPOSITORY}"
    fi
    git -C "${NCNN_SOURCE}" fetch --depth 1 origin "${NCNN_REF}"
    git -C "${NCNN_SOURCE}" checkout --detach FETCH_HEAD
fi

cmake -S "${NCNN_SOURCE}" -B "${NCNN_BUILD}" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="${NDK}/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM="android-${API}" \
    -DANDROID_STL=c++_static \
    -DNCNN_SHARED_LIB=OFF \
    -DNCNN_OPENMP=OFF \
    -DNCNN_VULKAN=OFF \
    -DNCNN_BUILD_BENCHMARK=OFF \
    -DNCNN_BUILD_EXAMPLES=OFF \
    -DNCNN_BUILD_TESTS=OFF \
    -DNCNN_BUILD_TOOLS=OFF

cmake --build "${NCNN_BUILD}" --target ncnn --parallel

export AR_aarch64_linux_android="${AR}"
export CC_aarch64_linux_android="${CLANG}"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="${CLANG}"
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER="/usr/bin/gcc"
export NCNN_LIB_DIR="${NCNN_BUILD}/src"

cargo build \
    --manifest-path "${ROOT}/ncnn_rust/Cargo.toml" \
    --target aarch64-linux-android \
    --release

mkdir -p "${OUTPUT_DIR}"
cp "${ROOT}/ncnn_rust/target/aarch64-linux-android/release/libncnn_rust.so" \
    "${OUTPUT_DIR}/libncnn_rust.so"
cp "${CXX_SHARED}" "${OUTPUT_DIR}/libc++_shared.so"
"${STRIP}" --strip-unneeded "${OUTPUT_DIR}/libncnn_rust.so"

echo "Android arm64-v8a SO: ${OUTPUT_DIR}/libncnn_rust.so"
