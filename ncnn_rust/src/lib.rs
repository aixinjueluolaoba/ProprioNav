#![allow(non_camel_case_types)]

use std::ffi::CStr;
use std::f32::consts::PI;
use std::os::raw::{c_char, c_void};
use std::time::Instant;

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
    fn ncnn_extractor_extract(
        ex: ncnn_extractor_t,
        name: *const c_char,
        mat: *mut ncnn_mat_t,
    ) -> i32;
    fn ncnn_mat_create() -> ncnn_mat_t;
    fn ncnn_mat_create_external_1d(
        w: i32,
        data: *mut c_void,
        allocator: *mut c_void,
    ) -> ncnn_mat_t;
    fn ncnn_mat_destroy(mat: ncnn_mat_t);
    fn ncnn_mat_get_data(mat: ncnn_mat_t) -> *mut f32;
}

// Default deployment scales (世界尺度/速度/定位年龄). 必须与模型训练环境一致;
// 运行时可经 `nav_configure` 覆盖, 所以同一份 .so 能服务不同尺度的游戏。
// 默认值 = V5 (世界 2250, 速度 500, 年龄 500/1000)。hidden 维从模型 param
// 自动识别 (见 detect_hidden_dim), 不再写死。
const WORLD_SIZE: f32 = 2250.0;
const DT: f32 = 0.3;
const OBS_DIM: i32 = 13;
const MAX_TURN: f32 = PI / 4.0;
const MAX_POSITION_AGE_MS: f32 = 500.0;
const MAX_ACCEPTED_AGE_MS: f32 = 2000.0;
const STALE_POSITION_AGE_MS: f32 = 1000.0;
const MIN_SAMPLE_DT: f32 = 0.05;
const MAX_SAMPLE_DT: f32 = 1.5;
const MAX_REASONABLE_SPEED: f32 = 500.0;

/// 每个会话的运行时可配尺度 (nav_configure 可覆盖)。
#[derive(Clone, Copy)]
struct NavConfig {
    world_size: f32,
    age_max_ms: f32,
    stale_ms: f32,
    max_speed: f32,
}

impl Default for NavConfig {
    fn default() -> Self {
        Self {
            world_size: WORLD_SIZE,
            age_max_ms: MAX_POSITION_AGE_MS,
            stale_ms: STALE_POSITION_AGE_MS,
            max_speed: MAX_REASONABLE_SPEED,
        }
    }
}

/// 从 NCNN param 文本里自动识别 LSTM hidden 维。
///
/// 导出图里有 `MemoryData b_hh ... 0=4*hidden`, 故 hidden = 0=/4。
/// 识别失败时回退到 96。
fn detect_hidden_dim(param_path: &str) -> usize {
    if let Ok(text) = std::fs::read_to_string(param_path) {
        for line in text.lines() {
            let mut it = line.split_whitespace();
            if it.next() == Some("MemoryData") {
                let _name = it.next();
                for tok in it {
                    if let Some(v) = tok.strip_prefix("0=") {
                        if let Ok(n) = v.parse::<usize>() {
                            if n % 4 == 0 && n >= 16 {
                                return n / 4;
                            }
                        }
                    }
                }
            }
        }
    }
    96
}
const JUMP_RETRY_COOLDOWN: i32 = 8;
const RECOVERY_COMMIT_STEPS: i32 = 4;
// 尺度自校准的“相对阈值”: 用自身近端步幅 (step_scale) 的比例判断移动/卡顿/碰撞,
// 而不是绝对距离 2.0 —— 这样慢速游戏不会误判卡住。
const REL_STUCK_RATIO: f32 = 0.15;    // 位移 < 15% 步幅 = 停住
const REL_VALID_RATIO: f32 = 0.15;    // 位移 >= 15% 步幅才采速度/朝向
const REL_COLLIDE_RATIO: f32 = 0.30;  // 位移 < 30% 步幅 = 被挡
// 防转圈: 大转向后锁步数 + 原地不动时禁止继续转。
const TURN_COOLDOWN_STEPS: i32 = 3;
const TURN_COOLDOWN_DEG: f32 = 30.0;
const TURN_MOVE_RATIO: f32 = 0.30;
const MACRO_TURNS: [f32; 8] = [
    0.0,
    -0.25 * PI,
    0.25 * PI,
    -0.5 * PI,
    0.5 * PI,
    -0.75 * PI,
    0.75 * PI,
    PI,
];
const MACRO_HOLD: [i32; 8] = [0, 4, 4, 6, 6, 6, 6, 8];
const TURN_OFFSETS: [f32; 7] = [
    -PI / 4.0,
    -25.0 * PI / 180.0,
    -10.0 * PI / 180.0,
    0.0,
    10.0 * PI / 180.0,
    25.0 * PI / 180.0,
    PI / 4.0,
];

#[inline]
fn normalize_angle(value: f32) -> f32 {
    value.sin().atan2(value.cos())
}

#[inline]
fn clamp(value: f32, low: f32, high: f32) -> f32 {
    value.max(low).min(high)
}

#[inline]
fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for index in 1..values.len() {
        if values[index] > values[best] {
            best = index;
        }
    }
    best
}

/// Load an NCNN model. Returns NULL on failure.
#[no_mangle]
pub unsafe extern "C" fn init_net(
    param_path: *const c_char,
    bin_path: *const c_char,
) -> *mut c_void {
    if param_path.is_null() || bin_path.is_null() {
        return std::ptr::null_mut();
    }
    let net = ncnn_net_create();
    if net.is_null() {
        return std::ptr::null_mut();
    }
    if ncnn_net_load_param(net, param_path) != 0 || ncnn_net_load_model(net, bin_path) != 0 {
        ncnn_net_destroy(net);
        return std::ptr::null_mut();
    }
    net
}

#[no_mangle]
pub unsafe extern "C" fn free_net(net: *mut c_void) {
    if !net.is_null() {
        ncnn_net_destroy(net);
    }
}

/// Low-level V4 inference.
///
/// Inputs: x[13], h_in[hidden], c_in[hidden]
/// Outputs: steer_logits[7], speed_logits[2], h_out[hidden], c_out[hidden]
unsafe fn run_inference_impl(
    net: *mut c_void,
    x: *const f32,
    h_in: *const f32,
    c_in: *const f32,
    steer_logits: *mut f32,
    speed_logits: *mut f32,
    h_out: *mut f32,
    c_out: *mut f32,
    macro_logits: *mut f32,
    hidden: usize,
) -> i32 {
    if net.is_null() || x.is_null() || h_in.is_null() || c_in.is_null() {
        return -1;
    }

    let extractor = ncnn_extractor_create(net);
    if extractor.is_null() {
        return -2;
    }
    let mat_x = ncnn_mat_create_external_1d(OBS_DIM, x as *mut c_void, std::ptr::null_mut());
    let mat_h = ncnn_mat_create_external_1d(hidden as i32, h_in as *mut c_void, std::ptr::null_mut());
    let mat_c = ncnn_mat_create_external_1d(hidden as i32, c_in as *mut c_void, std::ptr::null_mut());
    let in0 = b"in0\0".as_ptr() as *const c_char;
    let in1 = b"in1\0".as_ptr() as *const c_char;
    let in2 = b"in2\0".as_ptr() as *const c_char;
    ncnn_extractor_input(extractor, in0, mat_x);
    ncnn_extractor_input(extractor, in1, mat_h);
    ncnn_extractor_input(extractor, in2, mat_c);

    let out0 = b"out0\0".as_ptr() as *const c_char;
    let out1 = b"out1\0".as_ptr() as *const c_char;
    let out2 = b"out2\0".as_ptr() as *const c_char;
    let out3 = b"out3\0".as_ptr() as *const c_char;
    let out4 = b"out4\0".as_ptr() as *const c_char;
    let mut mat_out0 = ncnn_mat_create();
    let mut mat_out1 = ncnn_mat_create();
    let mut mat_out2 = ncnn_mat_create();
    let mut mat_out3 = ncnn_mat_create();
    let mut mat_out4 = ncnn_mat_create();
    let failed = ncnn_extractor_extract(extractor, out0, &mut mat_out0) != 0
        || ncnn_extractor_extract(extractor, out1, &mut mat_out1) != 0
        || ncnn_extractor_extract(extractor, out2, &mut mat_out2) != 0
        || ncnn_extractor_extract(extractor, out3, &mut mat_out3) != 0
        || (!macro_logits.is_null() && ncnn_extractor_extract(extractor, out4, &mut mat_out4) != 0);

    let result = if failed {
        -3
    } else {
        if !steer_logits.is_null() {
            std::ptr::copy_nonoverlapping(ncnn_mat_get_data(mat_out0), steer_logits, 7);
        }
        if !speed_logits.is_null() {
            std::ptr::copy_nonoverlapping(ncnn_mat_get_data(mat_out1), speed_logits, 2);
        }
        if !h_out.is_null() {
            std::ptr::copy_nonoverlapping(ncnn_mat_get_data(mat_out2), h_out, hidden);
        }
        if !c_out.is_null() {
            std::ptr::copy_nonoverlapping(ncnn_mat_get_data(mat_out3), c_out, hidden);
        }
        if !macro_logits.is_null() {
            std::ptr::copy_nonoverlapping(ncnn_mat_get_data(mat_out4), macro_logits, 8);
        }
        0
    };

    ncnn_mat_destroy(mat_x);
    ncnn_mat_destroy(mat_h);
    ncnn_mat_destroy(mat_c);
    ncnn_mat_destroy(mat_out0);
    ncnn_mat_destroy(mat_out1);
    ncnn_mat_destroy(mat_out2);
    ncnn_mat_destroy(mat_out3);
    ncnn_mat_destroy(mat_out4);
    ncnn_extractor_destroy(extractor);
    result
}

/// Backward-compatible low-level V4 inference ABI.
#[no_mangle]
pub unsafe extern "C" fn run_inference(
    net: *mut c_void,
    x: *const f32,
    h_in: *const f32,
    c_in: *const f32,
    steer_logits: *mut f32,
    speed_logits: *mut f32,
    h_out: *mut f32,
    c_out: *mut f32,
) -> i32 {
    run_inference_impl(
        net, x, h_in, c_in, steer_logits, speed_logits, h_out, c_out,
        std::ptr::null_mut(), 96,
    )
}

/// V5 low-level inference ABI with the macro logits head.
#[no_mangle]
pub unsafe extern "C" fn run_inference_macro(
    net: *mut c_void,
    x: *const f32,
    h_in: *const f32,
    c_in: *const f32,
    steer_logits: *mut f32,
    speed_logits: *mut f32,
    h_out: *mut f32,
    c_out: *mut f32,
    macro_logits: *mut f32,
) -> i32 {
    run_inference_impl(
        net, x, h_in, c_in, steer_logits, speed_logits, h_out, c_out, macro_logits, 96,
    )
}

fn freshness_scale(age_ms: f32, stale_ms: f32) -> f32 {
    // Must mirror GPUUnknownHeadingNavEnv._freshness_scale: 1.0 until 200 ms,
    // ramp to 0.5 at 500 ms, then ramp to 0 at the stale threshold.
    if age_ms <= 200.0 {
        1.0
    } else if age_ms <= 500.0 {
        1.0 - 0.5 * ((age_ms - 200.0) / 300.0)
    } else if age_ms <= stale_ms {
        0.5 * ((stale_ms - age_ms) / (stale_ms - 500.0))
    } else {
        0.0
    }
}

struct NavState {
    net: ncnn_net_t,
    hidden: usize,
    h: Vec<f32>,
    c: Vec<f32>,
    cfg: NavConfig,
    first_step: bool,
    old_pos: [f32; 2],
    old_age_ms: f32,
    last_receive_at: Instant,
    velocity: [f32; 2],
    estimated_heading: f32,
    heading_valid: bool,
    heading_confidence: f32,
    calibration_samples: i32,
    no_valid_observation_steps: i32,
    probe_phase: i32,
    last_turn_delta: f32,
    turn_history: [f32; 2],
    prev_dist: f32,
    prev_speed: f32,
    prev_jump: i32,
    time_since_collision: f32,
    stuck_time: f32,
    no_progress_time: f32,
    step_scale: f32,
    turn_cooldown: i32,
    jump_cooldown: i32,
    recovery_commit_left: i32,
    recovery_phase: i32,
    recovery_bin: usize,
    macro_left: i32,
    macro_target_heading: f32,
    active_macro: i32,
    /// 直接喂给摇杆的绝对世界朝向, 由库内部维护 (rad, y 向上, 0 = +x/右)。
    /// 首次调用用"当前点->目标"的方位角初始化, 之后累加策略的相对转角。
    joy_heading: f32,
    joy_heading_valid: bool,
}

impl NavState {
    fn new(net: ncnn_net_t, hidden: usize) -> Self {
        Self {
            net,
            hidden,
            h: vec![0.0; hidden],
            c: vec![0.0; hidden],
            cfg: NavConfig::default(),
            first_step: true,
            old_pos: [0.0; 2],
            old_age_ms: 0.0,
            last_receive_at: Instant::now(),
            velocity: [0.0; 2],
            estimated_heading: 0.0,
            heading_valid: false,
            heading_confidence: 0.0,
            calibration_samples: 0,
            no_valid_observation_steps: 0,
            probe_phase: 0,
            last_turn_delta: 0.0,
            turn_history: [0.0; 2],
            prev_dist: 0.0,
            prev_speed: 0.0,
            prev_jump: 0,
            time_since_collision: 10.0,
            stuck_time: 0.0,
            no_progress_time: 0.0,
            step_scale: 0.0,
            turn_cooldown: 0,
            jump_cooldown: 0,
            recovery_commit_left: 0,
            recovery_phase: 0,
            recovery_bin: 6,
            macro_left: 0,
            macro_target_heading: 0.0,
            active_macro: 0,
            joy_heading: 0.0,
            joy_heading_valid: false,
        }
    }
}

/// Initialize one navigation session. Target position is supplied to nav_step.
#[no_mangle]
pub unsafe extern "C" fn nav_init(
    param_path: *const c_char,
    bin_path: *const c_char,
) -> *mut c_void {
    let net = init_net(param_path, bin_path);
    if net.is_null() {
        return std::ptr::null_mut();
    }
    // hidden 维从模型 param 自动识别, 同一 .so 可加载 96 / 192 等不同模型。
    let hidden = if param_path.is_null() {
        96
    } else {
        match CStr::from_ptr(param_path).to_str() {
            Ok(p) => detect_hidden_dim(p),
            Err(_) => 96,
        }
    };
    Box::into_raw(Box::new(NavState::new(net, hidden))) as *mut c_void
}

/// 运行时可配的游戏尺度 (不重编即可适配不同地图/速度/时延)。
/// 传 <=0 的字段表示保持默认值不变。
#[no_mangle]
pub unsafe extern "C" fn nav_configure(
    nav: *mut c_void,
    world_size: f32,
    age_max_ms: f32,
    stale_ms: f32,
    max_speed: f32,
) -> i32 {
    if nav.is_null() {
        return -1;
    }
    let state = &mut *(nav as *mut NavState);
    if world_size > 0.0 {
        state.cfg.world_size = world_size;
    }
    if age_max_ms > 0.0 {
        state.cfg.age_max_ms = age_max_ms;
    }
    if stale_ms > 0.0 {
        state.cfg.stale_ms = stale_ms;
    }
    if max_speed > 0.0 {
        state.cfg.max_speed = max_speed;
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn nav_free(nav: *mut c_void) {
    if nav.is_null() {
        return;
    }
    let state = Box::from_raw(nav as *mut NavState);
    free_net(state.net);
}

/// V4 navigation API without a heading input.
///
/// Inputs: current position, fixed target position, position age in ms.
/// Outputs:
///   turn_delta  relative turn in radians ([-pi/4, pi/4]) from the previous
///               movement direction, for callers that accumulate it themselves.
///   abs_angle   absolute world-frame joystick/movement heading in radians
///               (y-up, 0 = +x/right, pi/2 = +y/up), maintained by the library;
///               seeded with the bearing to the target on the first call. Feed it
///               straight to the joystick and do NOT also accumulate turn_delta.
///   speed       freshness-limited speed [0, 100]
///   jump        0 or 1
/// Passing NULL for abs_angle skips that output.
/// Motion direction, collision, recurrent state, and calibration are internal.
#[no_mangle]
pub unsafe extern "C" fn nav_step(
    nav: *mut c_void,
    pos_x: f32,
    pos_y: f32,
    target_x: f32,
    target_y: f32,
    position_age_ms: f32,
    turn_delta_out: *mut f32,
    speed_out: *mut f32,
    jump_out: *mut i32,
    abs_angle_out: *mut f32,
) -> i32 {
    nav_step_internal(
        nav,
        pos_x,
        pos_y,
        target_x,
        target_y,
        position_age_ms,
        None,
        turn_delta_out,
        speed_out,
        jump_out,
        std::ptr::null_mut(),
        abs_angle_out,
    )
}

/// V5 feedback API. Collision is authoritative input from the game.
#[no_mangle]
pub unsafe extern "C" fn nav_step_feedback(
    nav: *mut c_void,
    pos_x: f32,
    pos_y: f32,
    target_x: f32,
    target_y: f32,
    position_age_ms: f32,
    collided: i32,
    turn_delta_out: *mut f32,
    speed_out: *mut f32,
    jump_out: *mut i32,
    macro_out: *mut i32,
    abs_angle_out: *mut f32,
) -> i32 {
    nav_step_internal(
        nav,
        pos_x,
        pos_y,
        target_x,
        target_y,
        position_age_ms,
        Some(collided != 0),
        turn_delta_out,
        speed_out,
        jump_out,
        macro_out,
        abs_angle_out,
    )
}

unsafe fn nav_step_internal(
    nav: *mut c_void,
    pos_x: f32,
    pos_y: f32,
    target_x: f32,
    target_y: f32,
    position_age_ms: f32,
    collided_override: Option<bool>,
    turn_delta_out: *mut f32,
    speed_out: *mut f32,
    jump_out: *mut i32,
    macro_out: *mut i32,
    abs_angle_out: *mut f32,
) -> i32 {
    if nav.is_null() || turn_delta_out.is_null() || speed_out.is_null() || jump_out.is_null() {
        return -1;
    }
    if !pos_x.is_finite()
        || !pos_y.is_finite()
        || !target_x.is_finite()
        || !target_y.is_finite()
        || !position_age_ms.is_finite()
        || position_age_ms < 0.0
    {
        return -4;
    }

    let state = &mut *(nav as *mut NavState);
    let age_ms = clamp(position_age_ms, 0.0, MAX_ACCEPTED_AGE_MS);
    let now = Instant::now();
    let macro_enabled = !macro_out.is_null();
    let mut collided = collided_override.unwrap_or(false);
    let mut collision_measurement_valid = false;
    let mut displacement = 0.0;
    let mut stalled = false;
    let mut sample_dt = DT;

    if !state.first_step {
        let elapsed = now.duration_since(state.last_receive_at).as_secs_f32();
        let control_dt = if (MIN_SAMPLE_DT..=MAX_SAMPLE_DT).contains(&elapsed) {
            elapsed
        } else {
            DT
        };
        sample_dt = control_dt - (age_ms - state.old_age_ms) / 1000.0;
        let moved_x = pos_x - state.old_pos[0];
        let moved_y = pos_y - state.old_pos[1];
        displacement = (moved_x * moved_x + moved_y * moved_y).sqrt();
        let sample_time_valid = (MIN_SAMPLE_DT..=MAX_SAMPLE_DT).contains(&sample_dt);
        collision_measurement_valid = sample_time_valid && age_ms <= state.cfg.stale_ms;
        // 用自身近端步幅做尺度基准 (冷启动用当前位移 bootstrap)。
        let scale0 = if state.step_scale > 0.0 {
            state.step_scale
        } else {
            displacement.max(1.0e-3)
        };
        stalled = displacement < REL_STUCK_RATIO * scale0;
        if collided_override.is_none() {
            collided = collision_measurement_valid && displacement < REL_COLLIDE_RATIO * scale0;
        } else {
            collision_measurement_valid = age_ms <= state.cfg.stale_ms;
        }
        // 正常移动时更新步幅 EMA (被挡/停住时不更新, 保留真实步幅基准)。
        if collision_measurement_valid && !collided && !stalled {
            state.step_scale = if state.step_scale > 0.0 {
                0.9 * state.step_scale + 0.1 * displacement
            } else {
                displacement
            };
        }
    }

    let sample_time_valid = (MIN_SAMPLE_DT..=MAX_SAMPLE_DT).contains(&sample_dt);
    let sample_speed = displacement / sample_dt.max(MIN_SAMPLE_DT);
    let scale_for_valid = if state.step_scale > 0.0 {
        state.step_scale
    } else {
        displacement
    };
    let velocity_valid = sample_time_valid
        && (state.step_scale <= 0.0 || displacement >= REL_VALID_RATIO * scale_for_valid)
        && sample_speed <= state.cfg.max_speed;
    if velocity_valid {
        let mut sample_velocity = [
            (pos_x - state.old_pos[0]) / sample_dt,
            (pos_y - state.old_pos[1]) / sample_dt,
        ];
        let sample_norm = (sample_velocity[0] * sample_velocity[0]
            + sample_velocity[1] * sample_velocity[1])
            .sqrt();
        if sample_norm > state.cfg.max_speed {
            let scale = state.cfg.max_speed / sample_norm;
            sample_velocity[0] *= scale;
            sample_velocity[1] *= scale;
        }
        let mut replay_turn = 0.0;
        if age_ms > DT * 1000.0 {
            replay_turn += state.turn_history[0];
        }
        if age_ms > DT * 2000.0 {
            replay_turn += state.turn_history[1];
        }
        let velocity_current = [
            sample_velocity[0] * replay_turn.cos() - sample_velocity[1] * replay_turn.sin(),
            sample_velocity[0] * replay_turn.sin() + sample_velocity[1] * replay_turn.cos(),
        ];
        state.velocity[0] = state.velocity[0] * 0.5 + velocity_current[0] * 0.5;
        state.velocity[1] = state.velocity[1] * 0.5 + velocity_current[1] * 0.5;

        let measured_heading = sample_velocity[1].atan2(sample_velocity[0]);
        let predicted_heading = normalize_angle(state.estimated_heading + state.last_turn_delta);
        let corrected_heading = normalize_angle(measured_heading + replay_turn);
        let heading_error = normalize_angle(corrected_heading - predicted_heading).abs();
        let first_measurement = !state.heading_valid;
        let consistent = heading_error <= 35.0 * PI / 180.0;
        state.estimated_heading = if first_measurement {
            corrected_heading
        } else {
            normalize_angle(
                predicted_heading
                    + normalize_angle(corrected_heading - predicted_heading) * 0.35,
            )
        };
        state.heading_valid = true;
        if first_measurement || consistent || velocity_valid {
            state.calibration_samples = (state.calibration_samples + 1).min(3);
            state.heading_confidence = if first_measurement || consistent {
                (state.heading_confidence + 0.25).min(1.0)
            } else {
                (state.heading_confidence - 0.15).max(0.0)
            };
        } else {
            state.heading_confidence = (state.heading_confidence - 0.15).max(0.0);
        }
        state.no_valid_observation_steps = 0;
    } else {
        state.estimated_heading = normalize_angle(state.estimated_heading + state.last_turn_delta);
        state.heading_confidence *= 0.98;
        state.no_valid_observation_steps += 1;
    }

    if !state.first_step {
        let prediction_age = age_ms.min(state.cfg.age_max_ms) / 1000.0;
        let predicted_x = pos_x + state.velocity[0] * prediction_age;
        let predicted_y = pos_y + state.velocity[1] * prediction_age;
        let predicted_dx = target_x - predicted_x;
        let predicted_dy = target_y - predicted_y;
        let distance = (predicted_dx * predicted_dx + predicted_dy * predicted_dy).sqrt();
        let progress = state.prev_dist - distance;

        state.time_since_collision = if collided {
            0.0
        } else {
            state.time_since_collision + DT
        };
        if collision_measurement_valid {
            state.stuck_time = if stalled {
                state.stuck_time + DT
            } else {
                0.0
            };
            state.no_progress_time = if progress < 0.2 {
                state.no_progress_time + DT
            } else {
                0.0
            };
        }
    }

    let prediction_age = age_ms.min(state.cfg.age_max_ms) / 1000.0;
    let predicted_x = clamp(pos_x + state.velocity[0] * prediction_age, -(state.cfg.world_size * 0.5 + 50.0), state.cfg.world_size * 0.5 + 50.0);
    let predicted_y = clamp(pos_y + state.velocity[1] * prediction_age, -(state.cfg.world_size * 0.5 + 50.0), state.cfg.world_size * 0.5 + 50.0);
    let dx = target_x - predicted_x;
    let dy = target_y - predicted_y;
    let distance = (dx * dx + dy * dy).sqrt();
    let target_angle = dy.atan2(dx);
    let angle_error = normalize_angle(target_angle - state.estimated_heading);
    let collision_touch = if state.time_since_collision < 0.6 { 1.0 } else { 0.0 };
    let jump_probe = if collided {
        if state.prev_jump != 0 { -1.0 } else { 1.0 }
    } else {
        0.0
    };
    let obs = [
        clamp(dx / state.cfg.world_size, -1.0, 1.0),
        clamp(dy / state.cfg.world_size, -1.0, 1.0),
        angle_error.sin(),
        angle_error.cos(),
        clamp(state.velocity[0] / state.cfg.max_speed, -1.0, 1.0),
        clamp(state.velocity[1] / state.cfg.max_speed, -1.0, 1.0),
        clamp(distance / state.cfg.world_size, 0.0, 1.0),
        clamp(state.stuck_time / 3.0, 0.0, 1.0),
        collision_touch,
        jump_probe,
        clamp(age_ms / state.cfg.age_max_ms, 0.0, 1.0),
        state.heading_confidence,
        clamp(state.last_turn_delta / MAX_TURN, -1.0, 1.0),
    ];

    let mut steer_logits = [0.0; 7];
    let mut speed_logits = [0.0; 2];
    let mut macro_logits = [0.0; 8];
    let mut h_out = vec![0.0f32; state.hidden];
    let mut c_out = vec![0.0f32; state.hidden];
    let inference_result = run_inference_impl(
        state.net,
        obs.as_ptr(),
        state.h.as_ptr(),
        state.c.as_ptr(),
        steer_logits.as_mut_ptr(),
        speed_logits.as_mut_ptr(),
        h_out.as_mut_ptr(),
        c_out.as_mut_ptr(),
        if macro_enabled { macro_logits.as_mut_ptr() } else { std::ptr::null_mut() },
        state.hidden,
    );
    if inference_result != 0 {
        return inference_result;
    }
    state.h = h_out;
    state.c = c_out;

    let mut steer_bin = argmax(&steer_logits);
    let mut speed_bin = argmax(&speed_logits);
    let calibrating = state.calibration_samples < 2;
    if calibrating {
        speed_bin = 0;
        if state.no_valid_observation_steps >= 2 {
            steer_bin = if state.probe_phase == 0 { 5 } else { 1 };
            state.probe_phase = 1 - state.probe_phase;
        } else {
            steer_bin = 3;
        }
    }
    let recovery_signal = collided
        || state.time_since_collision < 0.9
        || state.no_progress_time > 0.6;
    if !macro_enabled && !calibrating && recovery_signal && state.recovery_commit_left <= 0 {
        state.recovery_bin = if state.recovery_phase == 0 { 6 } else { 0 };
        state.recovery_phase = 1 - state.recovery_phase;
        state.recovery_commit_left = RECOVERY_COMMIT_STEPS;
    }
    if !macro_enabled && !calibrating && state.recovery_commit_left > 0 {
        steer_bin = state.recovery_bin;
        state.recovery_commit_left -= 1;
    }
    let mut turn_delta = TURN_OFFSETS[steer_bin];
    if macro_enabled && !calibrating {
        let macro_bin = argmax(&macro_logits) as i32;
        if state.macro_left <= 0 && macro_bin > 0 {
            state.active_macro = macro_bin;
            state.macro_target_heading = normalize_angle(
                state.estimated_heading + MACRO_TURNS[macro_bin as usize],
            );
            state.macro_left = MACRO_HOLD[macro_bin as usize];
        }
        if state.macro_left > 0 {
            let macro_error = normalize_angle(
                state.macro_target_heading - state.estimated_heading,
            );
            turn_delta = clamp(macro_error, -MAX_TURN, MAX_TURN);
            state.macro_left -= 1;
        } else {
            state.active_macro = 0;
        }
    }
    // 防转圈: 非标定/未碰撞/非脱困/无宏锁定时, 给策略转向加"冷却 + 移动门槛",
    // 避免每步打满 ±45° 原地打转。
    if !calibrating && !collided && state.recovery_commit_left == 0 && state.active_macro == 0 {
        let scale = if state.step_scale > 0.0 {
            state.step_scale
        } else {
            f32::INFINITY
        };
        let moving = state.first_step || displacement >= TURN_MOVE_RATIO * scale;
        if state.turn_cooldown > 0 {
            turn_delta = 0.0;
            state.turn_cooldown -= 1;
        } else if !moving {
            turn_delta = 0.0;
        } else if turn_delta.abs() >= TURN_COOLDOWN_DEG.to_radians() {
            state.turn_cooldown = TURN_COOLDOWN_STEPS;
        }
    }

    let base_speed = if speed_bin == 1 { 100.0 } else { 50.0 };
    let speed = base_speed * freshness_scale(age_ms, state.cfg.stale_ms);

    if !collided {
        state.jump_cooldown = 0;
    } else if state.prev_jump != 0 {
        state.jump_cooldown = JUMP_RETRY_COOLDOWN;
    } else if state.jump_cooldown > 0 {
        state.jump_cooldown -= 1;
    }
    let jump = if !calibrating && collided && state.prev_jump == 0 && state.jump_cooldown == 0 {
        1
    } else {
        0
    };

    // 绝对摇杆朝向 (库内部维护): 首次调用直接用"当前点->目标"的方位角, 让角色
    // 起手就朝目标, 免去开头试探; 之后累加策略/脱困给出的相对转角。输出即调用方
    // 直接写进摇杆的世界绝对角, 调用方不要再自己累加 turn_delta。
    // 注意种子用**原始两点**算, 不用带预测补偿的 target_angle: 首步速度估计尚未
    // 可信, 用它会把方位角带偏。
    if !state.joy_heading_valid {
        state.joy_heading = normalize_angle((target_y - pos_y).atan2(target_x - pos_x));
        state.joy_heading_valid = true;
    } else {
        state.joy_heading = normalize_angle(state.joy_heading + turn_delta);
    }

    *turn_delta_out = turn_delta;
    *speed_out = speed;
    *jump_out = jump;
    if !macro_out.is_null() {
        *macro_out = state.active_macro;
    }
    if !abs_angle_out.is_null() {
        *abs_angle_out = state.joy_heading;
    }
    state.old_pos = [pos_x, pos_y];
    state.old_age_ms = age_ms;
    state.last_receive_at = now;
    state.prev_speed = speed;
    state.prev_jump = jump;
    state.last_turn_delta = turn_delta;
    state.turn_history[1] = state.turn_history[0];
    state.turn_history[0] = turn_delta;
    state.first_step = false;
    state.prev_dist = distance;
    0
}
