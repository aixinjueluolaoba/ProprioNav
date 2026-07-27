fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap();
    let default_build_dir = if target_os == "android" {
        "build-android-arm64"
    } else {
        "build"
    };
    let ncnn_lib_dir = std::env::var_os("NCNN_LIB_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(&manifest_dir)
                .parent()
                .unwrap()
                .join("pipeline_out")
                .join("ncnn_source")
                .join(default_build_dir)
                .join("src")
        });

    println!("cargo:rerun-if-env-changed=NCNN_LIB_DIR");
    println!("cargo:rustc-link-search=native={}", ncnn_lib_dir.display());
    println!("cargo:rustc-link-lib=static=ncnn");

    if target_os == "android" {
        // Android 构建使用 NCNN_OPENMP=OFF；libc++_shared 随 AAR 一起打包。
        println!("cargo:rustc-link-lib=dylib=c++_shared");
        println!("cargo:rustc-link-lib=dylib=android");
        println!("cargo:rustc-link-lib=dylib=jnigraphics");
        println!("cargo:rustc-link-lib=dylib=log");
        println!("cargo:rustc-link-lib=dylib=m");
        println!("cargo:rustc-link-lib=dylib=dl");
        println!("cargo:rustc-link-arg=-Wl,--exclude-libs,ALL");
    } else {
        println!("cargo:rustc-link-lib=dylib=gomp");
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-link-lib=dylib=m");
    }
}
