plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

val generatedModelAssets = layout.buildDirectory.dir("generated/modelAssets")
val prepareModelAssets by tasks.registering(Copy::class) {
    from(rootProject.projectDir.parentFile.resolve("pipeline_out/policy_v4.param"))
    from(rootProject.projectDir.parentFile.resolve("pipeline_out/policy_v4.bin"))
    into(generatedModelAssets.map { it.dir("proprionav") })
}

android {
    namespace = "com.proprionav"
    compileSdk = 34
    ndkVersion = "26.1.10909125"

    defaultConfig {
        minSdk = 24
        consumerProguardFiles("consumer-rules.pro")
        ndk {
            abiFilters += "arm64-v8a"
        }
        externalNativeBuild {
            cmake {
                cppFlags += listOf("-std=c++17", "-fexceptions", "-frtti")
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    sourceSets["main"].assets.srcDir(generatedModelAssets)

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

tasks.named("preBuild").configure {
    dependsOn(prepareModelAssets)
}
