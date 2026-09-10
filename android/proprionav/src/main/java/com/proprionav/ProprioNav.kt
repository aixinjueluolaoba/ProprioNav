package com.proprionav

import android.content.Context
import java.io.File

data class NavOutput(
    val turnDelta: Float,
    val speed: Float,
    val jump: Boolean,
)

class ProprioNav private constructor(private var handle: Long) : AutoCloseable {
    @Synchronized
    fun step(
        posX: Float,
        posY: Float,
        targetX: Float,
        targetY: Float,
        positionAgeMs: Float,
    ): NavOutput {
        check(handle != 0L) { "ProprioNav session is closed" }
        val values = nativeStep(handle, posX, posY, targetX, targetY, positionAgeMs)
        return NavOutput(
            turnDelta = values[0],
            speed = values[1],
            jump = values[2] >= 0.5F,
        )
    }

    @Synchronized
    override fun close() {
        if (handle != 0L) {
            nativeDestroy(handle)
            handle = 0L
        }
    }

    companion object {
        private const val ASSET_DIR = "proprionav"
        private const val PARAM = "policy_v4.param"
        private const val BIN = "policy_v4.bin"

        init {
            System.loadLibrary("ncnn_rust")
            System.loadLibrary("proprionav_jni")
        }

        @JvmStatic
        fun fromAssets(context: Context): ProprioNav {
            val modelDir = File(context.filesDir, "proprionav/v3")
            val param = copyAsset(context, "$ASSET_DIR/$PARAM", File(modelDir, PARAM))
            val bin = copyAsset(context, "$ASSET_DIR/$BIN", File(modelDir, BIN))
            val handle = nativeCreate(param.absolutePath, bin.absolutePath)
            check(handle != 0L) { "Failed to initialize ProprioNav" }
            return ProprioNav(handle)
        }

        private fun copyAsset(context: Context, asset: String, output: File): File {
            output.parentFile?.mkdirs()
            val temporary = File(output.parentFile, "${output.name}.tmp")
            context.assets.open(asset).use { input ->
                temporary.outputStream().use { destination ->
                    input.copyTo(destination)
                }
            }
            check(temporary.renameTo(output) || temporary.copyTo(output, overwrite = true).let {
                temporary.delete()
                true
            }) { "Failed to copy model asset: $asset" }
            return output
        }

        @JvmStatic
        private external fun nativeCreate(paramPath: String, binPath: String): Long

        @JvmStatic
        private external fun nativeStep(
            handle: Long,
            posX: Float,
            posY: Float,
            targetX: Float,
            targetY: Float,
            positionAgeMs: Float,
        ): FloatArray

        @JvmStatic
        private external fun nativeDestroy(handle: Long)
    }
}
