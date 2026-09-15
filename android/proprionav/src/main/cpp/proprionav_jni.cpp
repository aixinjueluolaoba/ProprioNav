#include <jni.h>

#include <cstdint>

#include "proprionav.h"

namespace {

void throw_illegal_state(JNIEnv* env, const char* message) {
    jclass cls = env->FindClass("java/lang/IllegalStateException");
    if (cls != nullptr) {
        env->ThrowNew(cls, message);
    }
}

void* from_handle(jlong handle) {
    return reinterpret_cast<void*>(static_cast<intptr_t>(handle));
}

jlong to_handle(void* value) {
    return static_cast<jlong>(reinterpret_cast<intptr_t>(value));
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_proprionav_ProprioNav_nativeCreate(
    JNIEnv* env,
    jclass,
    jstring param_path,
    jstring bin_path
) {
    if (param_path == nullptr || bin_path == nullptr) {
        throw_illegal_state(env, "Model paths must not be null");
        return 0;
    }
    const char* param = env->GetStringUTFChars(param_path, nullptr);
    const char* bin = env->GetStringUTFChars(bin_path, nullptr);
    if (param == nullptr || bin == nullptr) {
        if (param != nullptr) {
            env->ReleaseStringUTFChars(param_path, param);
        }
        if (bin != nullptr) {
            env->ReleaseStringUTFChars(bin_path, bin);
        }
        return 0;
    }

    void* nav = nav_init(param, bin);
    env->ReleaseStringUTFChars(param_path, param);
    env->ReleaseStringUTFChars(bin_path, bin);
    if (nav == nullptr) {
        throw_illegal_state(env, "nav_init failed to load the NCNN model");
        return 0;
    }
    return to_handle(nav);
}

extern "C" JNIEXPORT void JNICALL
Java_com_proprionav_ProprioNav_nativeConfigure(
    JNIEnv* env,
    jclass,
    jlong handle,
    jfloat world_size,
    jfloat age_max_ms,
    jfloat stale_ms,
    jfloat max_speed
) {
    void* nav = from_handle(handle);
    if (nav == nullptr) {
        throw_illegal_state(env, "ProprioNav session is closed");
        return;
    }
    const int status = nav_configure(nav, world_size, age_max_ms, stale_ms, max_speed);
    if (status != 0) {
        throw_illegal_state(env, "nav_configure failed");
    }
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_proprionav_ProprioNav_nativeStep(
    JNIEnv* env,
    jclass,
    jlong handle,
    jfloat pos_x,
    jfloat pos_y,
    jfloat target_x,
    jfloat target_y,
    jfloat position_age_ms
) {
    void* nav = from_handle(handle);
    if (nav == nullptr) {
        throw_illegal_state(env, "ProprioNav session is closed");
        return nullptr;
    }

    float turn_delta = 0.0F;
    float speed = 0.0F;
    int jump = 0;
    float abs_angle = 0.0F;
    const int status = nav_step(
        nav,
        pos_x,
        pos_y,
        target_x,
        target_y,
        position_age_ms,
        &turn_delta,
        &speed,
        &jump,
        &abs_angle
    );
    if (status != 0) {
        throw_illegal_state(env, "nav_step failed");
        return nullptr;
    }

    const jfloat values[] = {
        turn_delta,
        speed,
        static_cast<jfloat>(jump),
        abs_angle,
    };
    jfloatArray result = env->NewFloatArray(4);
    if (result != nullptr) {
        env->SetFloatArrayRegion(result, 0, 4, values);
    }
    return result;
}

extern "C" JNIEXPORT void JNICALL
Java_com_proprionav_ProprioNav_nativeDestroy(
    JNIEnv*,
    jclass,
    jlong handle
) {
    void* nav = from_handle(handle);
    if (nav != nullptr) {
        nav_free(nav);
    }
}
