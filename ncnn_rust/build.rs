fn main() {
    // 获取当前 Cargo 工程根目录，动态拼接并定位静态库 libncnn.a 路径
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let ncnn_lib_dir = std::path::Path::new(&manifest_dir)
        .parent()
        .unwrap()
        .join("pipeline_out")
        .join("ncnn_source")
        .join("build")
        .join("src");
    
    println!("cargo:rustc-link-search=native={}", ncnn_lib_dir.display());
    
    // 静态链接 ncnn
    println!("cargo:rustc-link-lib=static=ncnn");
    
    // 动态链接编译依赖的 GNU OpenMP、C++ 标准库和数学库
    println!("cargo:rustc-link-lib=dylib=gomp");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=m");
}
