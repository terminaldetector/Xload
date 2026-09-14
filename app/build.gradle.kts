plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.kotlin.serialization)
    id("com.chaquo.python") version "17.0.0"
}

android {
    namespace = "io.github.terminaldetector.xload.app"
    compileSdk = 35

    defaultConfig {
        applicationId = "io.github.terminaldetector.xload"
        minSdk = 30
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"

        ndk {
            // Chaquopy ships a full CPython build per ABI; keep this to the ABIs
            // that matter for real devices + the emulator to limit APK size.
            abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86_64")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures {
        compose = true
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

// Chaquopy 17.x configures Python through its own top-level `chaquopy { }`
// extension rather than nesting inside `android { }`. Confirmed against the
// plugin's own demo app (github.com/chaquo/chaquopy/blob/master/demo/app/build.gradle.kts)
// since chaquo.com itself is unreachable from this sandbox to check the docs
// directly. The older android.defaultConfig.python{} / sourceSets.python.srcDirs()
// nesting this file used before doesn't exist in current versions and fails
// Kotlin DSL compilation ("Unresolved reference: python") -- caught by the
// first real GitHub Actions build, since :app was never built end to end here.
chaquopy {
    defaultConfig {
        version = "3.11"
        pip {
            // Pinned deliberately: termux-train 1.1.4/1.1.5 added a hard
            // dependency on `ameva-component-sdk`, which is not published
            // on PyPI (verified 404) and makes those versions uninstallable.
            // 1.1.3 is the newest version that installs and runs cleanly —
            // verified end-to-end (LoRA injection, training, SafeTensors
            // checkpoint round-trip) in a local venv against this exact pin.
            install("termux-train==1.1.3")
            // gguf: the llama.cpp project's own reference GGUF reader and
            // (critically) dequantize() for every GGML quant type -- used
            // instead of hand-rolling quantization math. Pure-Python wheel,
            // so it doesn't need a per-ABI native build like most of the
            // ML ecosystem does. numpy is its hard dependency and also
            // unlocks termux-train's own faster "accelerated" backend.
            install("gguf==0.19.0")
            install("numpy>=1.20.0")
        }
    }
}

chaquopy.sourceSets.getByName("main") {
    srcDir("src/main/python")
}

dependencies {
    implementation(project(":core"))

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.viewmodel.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.ui)
    implementation(libs.androidx.ui.graphics)
    implementation(libs.androidx.ui.tooling.preview)
    implementation(libs.androidx.material3)
    implementation(libs.androidx.navigation.compose)
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.kotlinx.serialization.json)

    debugImplementation(libs.androidx.ui.tooling)

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.test.ext.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.ui.test.junit4)
    debugImplementation(libs.androidx.ui.test.manifest)
}
