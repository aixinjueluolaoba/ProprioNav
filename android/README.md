# Android 接入

当前 Android 版本支持 `arm64-v8a`、Android 7.0（API 24）及以上。`proprionav`
模块生成可直接接入其他应用的 AAR，`sample` 模块是最小可运行示例。
NCNN、Rust SO、JNI、模型与 `libc++_shared.so` 均已打包进 AAR。

## 构建

```bash
# 1. 交叉编译无 OpenMP 的 NCNN + Rust 导航 SO
export ANDROID_NDK_HOME="$ANDROID_HOME/ndk/26.1.10909125"
bash scripts/build_android_arm64.sh

# 2. 构建 AAR 和示例 APK
cd android
./gradlew :proprionav:assembleRelease :sample:assembleDebug
```

脚本默认固定 NCNN commit `d0d50631179e380e587fb188338f5683d2c93275`，
可通过 `NCNN_REF` 与 `NCNN_REPOSITORY` 覆盖。

产物：

- `proprionav/build/outputs/aar/proprionav-release.aar`
- `sample/build/outputs/apk/debug/sample-debug.apk`

仓库同时保留了已验证产物，可直接使用：

- `dist/proprionav-v3-arm64.aar`
- `dist/proprionav-sample-arm64-debug.apk`
- `dist/SHA256SUMS`

## Kotlin 调用

把 AAR 加入应用依赖后：

```kotlin
val nav = ProprioNav.fromAssets(context)

val action = nav.step(
    posX = playerX,
    posY = playerY,
    targetX = goalX,
    targetY = goalY,
    heading = playerHeadingRadians,
)

moveDirection = action.direction
moveSpeed = action.speed
if (action.jump) {
    jump()
}

// 导航结束或页面销毁时
nav.close()
```

每个角色或导航任务应持有独立的 `ProprioNav` 实例。每个决策周期必须传入执行上一条
指令后的最新实际位置；碰撞、停滞、LSTM 和记忆状态全部由 SO 内部维护。
