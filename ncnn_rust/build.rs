fn main() {
    // 指定静态库 libncnn.a 的构建搜索目录
    println!("cargo:rustc-link-search=native=/home/diana/盲人寻路/pipeline_out/ncnn_source/build/src");
    
    // 静态链接 ncnn
    println!("cargo:rustc-link-lib=static=ncnn");
    
    // 动态链接编译依赖的 GNU OpenMP、C++ 标准库和数学库
    println!("cargo:rustc-link-lib=dylib=gomp");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=m");
}
