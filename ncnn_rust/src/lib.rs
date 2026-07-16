use std::ffi::{c_char, c_void};

// NCNN 不透明指针类型
pub type ncnn_net_t = *mut c_void;
pub type ncnn_extractor_t = *mut c_void;
pub type ncnn_mat_t = *mut c_void;

#[link(name = "ncnn", kind = "static")]
extern "C" {
    fn ncnn_net_create() -> ncnn_net_t;
    fn ncnn_net_destroy(net: ncnn_net_t);
    fn ncnn_net_load_param(net: ncnn_net_t, path: *const c_char) -> i32;
    fn ncnn_net_load_model(net: ncnn_net_t, path: *const c_char) -> i32;

    fn ncnn_extractor_create(net: ncnn_net_t) -> ncnn_extractor_t;
    fn ncnn_extractor_destroy(ex: ncnn_extractor_t);
    fn ncnn_extractor_input(ex: ncnn_extractor_t, name: *const c_char, mat: ncnn_mat_t) -> i32;
    fn ncnn_extractor_extract(ex: ncnn_extractor_t, name: *const c_char, mat: *mut ncnn_mat_t) -> i32;

    fn ncnn_mat_create() -> ncnn_mat_t;
    fn ncnn_mat_create_external_1d(w: i32, data: *mut c_void, allocator: *mut c_void) -> ncnn_mat_t;
    fn ncnn_mat_destroy(mat: ncnn_mat_t);
    fn ncnn_mat_get_data(mat: ncnn_mat_t) -> *mut f32;
}

/// 初始化网络结构与权重，成功则返回 Net 句柄，失败返回 NULL
#[no_mangle]
pub unsafe extern "C" fn init_net(param_path: *const c_char, bin_path: *const c_char) -> *mut c_void {
    let net = ncnn_net_create();
    if net.is_null() {
        return std::ptr::null_mut();
    }
    
    let ret_param = ncnn_net_load_param(net, param_path);
    if ret_param != 0 {
        ncnn_net_destroy(net);
        return std::ptr::null_mut();
    }
    
    let ret_model = ncnn_net_load_model(net, bin_path);
    if ret_model != 0 {
        ncnn_net_destroy(net);
        return std::ptr::null_mut();
    }
    
    net as *mut c_void
}

/// 释放网络
#[no_mangle]
pub unsafe extern "C" fn free_net(net_ptr: *mut c_void) {
    if !net_ptr.is_null() {
        ncnn_net_destroy(net_ptr as ncnn_net_t);
    }
}

/// 单步推理计算
/// 输入：
/// - net_ptr: 实例句柄
/// - x: float[12] 观测向量
/// - h_in: float[64] 输入隐藏状态
/// - c_in: float[64] 输入细胞状态
/// 输出：
/// - steer_logits: float[5]
/// - speed_logits: float[2]
/// - macro_logits: float[8]
/// - h_out: float[64] 传入传出以保存状态更新
/// - c_out: float[64]
#[no_mangle]
pub unsafe extern "C" fn run_inference(
    net_ptr: *mut c_void,
    x: *const f32,
    h_in: *const f32,
    c_in: *const f32,
    steer_logits: *mut f32,
    speed_logits: *mut f32,
    macro_logits: *mut f32,
    h_out: *mut f32,
    c_out: *mut f32,
) -> i32 {
    if net_ptr.is_null() || x.is_null() || h_in.is_null() || c_in.is_null() {
        return -1;
    }
    
    let net = net_ptr as ncnn_net_t;
    let ex = ncnn_extractor_create(net);
    if ex.is_null() {
        return -2;
    }
    
    // 构造 NCNN 零拷贝输入数据 Mat
    let mat_x = ncnn_mat_create_external_1d(12, x as *mut c_void, std::ptr::null_mut());
    let mat_h = ncnn_mat_create_external_1d(64, h_in as *mut c_void, std::ptr::null_mut());
    let mat_c = ncnn_mat_create_external_1d(64, c_in as *mut c_void, std::ptr::null_mut());
    
    let c_in0 = std::ffi::CString::new("in0").unwrap();
    let c_in1 = std::ffi::CString::new("in1").unwrap();
    let c_in2 = std::ffi::CString::new("in2").unwrap();
    
    ncnn_extractor_input(ex, c_in0.as_ptr(), mat_x);
    ncnn_extractor_input(ex, c_in1.as_ptr(), mat_h);
    ncnn_extractor_input(ex, c_in2.as_ptr(), mat_c);
    
    let c_out0 = std::ffi::CString::new("out0").unwrap();
    let c_out1 = std::ffi::CString::new("out1").unwrap();
    let c_out2 = std::ffi::CString::new("out2").unwrap();
    let c_out3 = std::ffi::CString::new("out3").unwrap();
    let c_out4 = std::ffi::CString::new("out4").unwrap();
    
    let mut mat_out0 = ncnn_mat_create();
    let mut mat_out1 = ncnn_mat_create();
    let mut mat_out2 = ncnn_mat_create();
    let mut mat_out3 = ncnn_mat_create();
    let mut mat_out4 = ncnn_mat_create();
    
    let mut ret = 0;
    
    // 执行特征提取提取五个分支的输出
    if ncnn_extractor_extract(ex, c_out0.as_ptr(), &mut mat_out0) != 0 ||
       ncnn_extractor_extract(ex, c_out1.as_ptr(), &mut mat_out1) != 0 ||
       ncnn_extractor_extract(ex, c_out2.as_ptr(), &mut mat_out2) != 0 ||
       ncnn_extractor_extract(ex, c_out3.as_ptr(), &mut mat_out3) != 0 ||
       ncnn_extractor_extract(ex, c_out4.as_ptr(), &mut mat_out4) != 0 
    {
        ret = -3;
    } else {
        // 复制推理结果
        if !steer_logits.is_null() {
            let data = ncnn_mat_get_data(mat_out0);
            std::ptr::copy_nonoverlapping(data, steer_logits, 5);
        }
        if !speed_logits.is_null() {
            let data = ncnn_mat_get_data(mat_out1);
            std::ptr::copy_nonoverlapping(data, speed_logits, 2);
        }
        if !macro_logits.is_null() {
            let data = ncnn_mat_get_data(mat_out2);
            std::ptr::copy_nonoverlapping(data, macro_logits, 8);
        }
        if !h_out.is_null() {
            let data = ncnn_mat_get_data(mat_out3);
            std::ptr::copy_nonoverlapping(data, h_out, 64);
        }
        if !c_out.is_null() {
            let data = ncnn_mat_get_data(mat_out4);
            std::ptr::copy_nonoverlapping(data, c_out, 64);
        }
    }
    
    // 清理释放 C 内存结构
    ncnn_mat_destroy(mat_x);
    ncnn_mat_destroy(mat_h);
    ncnn_mat_destroy(mat_c);
    
    ncnn_mat_destroy(mat_out0);
    ncnn_mat_destroy(mat_out1);
    ncnn_mat_destroy(mat_out2);
    ncnn_mat_destroy(mat_out3);
    ncnn_mat_destroy(mat_out4);
    
    ncnn_extractor_destroy(ex);
    
    ret
}
