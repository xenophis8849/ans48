#!/usr/bin/env bash
# 编 Android 库：prebuilt/<ABI>/libans48.so（arm64-v8a、armeabi-v7a），另编手机上跑的自检程序 build-android-<ABI>/ans48_check。
# 需要 NDK（r26 以上；本仓库的预编译包用 r28）：ANDROID_NDK=/path/to/ndk tools/build_android.sh
# 另给 ANS48_AEC48_DIR=<aec48 目录> 时，顺带编与 aec48 串联核对的 ans48_aecchain（链 aec48 的预编译库，只用于测试）
set -eu; cd "$(dirname "$0")/.."; NDK=${ANDROID_NDK:?请设 ANDROID_NDK}
for abi in arm64-v8a armeabi-v7a; do
  cmake -S . -B build-android-$abi -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=$abi -DANDROID_PLATFORM=android-21 \
    -DANDROID_STL=c++_static -DCMAKE_BUILD_TYPE=Release -DANS48_BUILD_CHECK=ON ${ANS48_AEC48_DIR:+-DANS48_AEC48_DIR=$ANS48_AEC48_DIR} > /dev/null
  cmake --build build-android-$abi -j4 > /dev/null
  mkdir -p prebuilt/$abi; cp build-android-$abi/libans48.so prebuilt/$abi/
  echo "$abi：prebuilt/$abi/libans48.so（$(stat -c %s prebuilt/$abi/libans48.so) 字节）"
done
