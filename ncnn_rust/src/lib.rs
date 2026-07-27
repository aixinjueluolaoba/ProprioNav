#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_void};
use std::f32::consts::PI;

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

const WORLD_SIZE: f32 = 2250.0;
const DT: f32 = 0.3;
const MAX_TURN: f32 = PI / 4.0;
const JUMP_RETRY_COOLDOWN: i32 = 8;
const HEADING_OFFSETS: [f32; 7] = [
    -PI / 4.0,
    -25.0 * PI / 180.0,
    -10.0 * PI / 180.0,
    0.0,
    10.0 * PI / 180.0,
    25.0 * PI / 180.0,
    PI / 4.0,
];
const GOAL_OFFSETS: [f32; 7] = [
    -10.0 * PI / 180.0,
    -6.0 * PI / 180.0,
    -2.5 * PI / 180.0,
    0.0,
    2.5 * PI / 180.0,
    6.0 * PI / 180.0,
    10.0 * PI / 180.0,
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

/// Low-level V3 inference.
///
/// Inputs: x[10], h_in[96], c_in[96]
/// Outputs: steer_logits[7], speed_logits[2], h_out[96], c_out[96]
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
    if net.is_null() || x.is_null() || h_in.is_null() || c_in.is_null() {
        return -1;
    }

    let extractor = ncnn_extractor_create(net);
    if extractor.is_null() {
        return -2;
    }
    let mat_x = ncnn_mat_create_external_1d(10, x as *mut c_void, std::ptr::null_mut());
    let mat_h = ncnn_mat_create_external_1d(96, h_in as *mut c_void, std::ptr::null_mut());
    let mat_c = ncnn_mat_create_external_1d(96, c_in as *mut c_void, std::ptr::null_mut());
    let in0 = c"in0";
    let in1 = c"in1";
    let in2 = c"in2";
    ncnn_extractor_input(extractor, in0.as_ptr(), mat_x);
    ncnn_extractor_input(extractor, in1.as_ptr(), mat_h);
    ncnn_extractor_input(extractor, in2.as_ptr(), mat_c);

    let out0 = c"out0";
    let out1 = c"out1";
    let out2 = c"out2";
    let out3 = c"out3";
    let mut mat_out0 = ncnn_mat_create();
    let mut mat_out1 = ncnn_mat_create();
    let mut mat_out2 = ncnn_mat_create();
    let mut mat_out3 = ncnn_mat_create();
    let failed = ncnn_extractor_extract(extractor, out0.as_ptr(), &mut mat_out0) != 0
        || ncnn_extractor_extract(extractor, out1.as_ptr(), &mut mat_out1) != 0
        || ncnn_extractor_extract(extractor, out2.as_ptr(), &mut mat_out2) != 0
        || ncnn_extractor_extract(extractor, out3.as_ptr(), &mut mat_out3) != 0;

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
            std::ptr::copy_nonoverlapping(ncnn_mat_get_data(mat_out2), h_out, 96);
        }
        if !c_out.is_null() {
            std::ptr::copy_nonoverlapping(ncnn_mat_get_data(mat_out3), c_out, 96);
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
    ncnn_extractor_destroy(extractor);
    result
}

struct NavState {
    net: ncnn_net_t,
    h: [f32; 96],
    c: [f32; 96],
    first_step: bool,
    old_pos: [f32; 2],
    prev_dist: f32,
    prev_speed: f32,
    prev_jump: i32,
    time_since_collision: f32,
    stuck_time: f32,
    no_progress_time: f32,
    jump_cooldown: i32,
}

impl NavState {
    fn new(net: ncnn_net_t) -> Self {
        Self {
            net,
            h: [0.0; 96],
            c: [0.0; 96],
            first_step: true,
            old_pos: [0.0; 2],
            prev_dist: 0.0,
            prev_speed: 0.0,
            prev_jump: 0,
            time_since_collision: 10.0,
            stuck_time: 0.0,
            no_progress_time: 0.0,
            jump_cooldown: 0,
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
    Box::into_raw(Box::new(NavState::new(net))) as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn nav_free(nav: *mut c_void) {
    if nav.is_null() {
        return;
    }
    let state = Box::from_raw(nav as *mut NavState);
    free_net(state.net);
}

/// Simplified navigation API.
///
/// Inputs: current position, target position, current heading (radians).
/// Outputs: movement direction (radians), speed (50/100), jump (0/1).
/// Collision, recurrent state, hybrid recovery, and jump probing are internal.
#[no_mangle]
pub unsafe extern "C" fn nav_step(
    nav: *mut c_void,
    pos_x: f32,
    pos_y: f32,
    target_x: f32,
    target_y: f32,
    heading: f32,
    direction_out: *mut f32,
    speed_out: *mut f32,
    jump_out: *mut i32,
) -> i32 {
    if nav.is_null() || direction_out.is_null() || speed_out.is_null() || jump_out.is_null() {
        return -1;
    }
    let state = &mut *(nav as *mut NavState);
    let dx = target_x - pos_x;
    let dy = target_y - pos_y;
    let distance = (dx * dx + dy * dy).sqrt();
    let target_angle = dy.atan2(dx);
    let angle_error = normalize_angle(target_angle - heading);

    let mut collided = false;
    if !state.first_step {
        let moved_x = pos_x - state.old_pos[0];
        let moved_y = pos_y - state.old_pos[1];
        let displacement = (moved_x * moved_x + moved_y * moved_y).sqrt();
        let progress = state.prev_dist - distance;
        let expected = state.prev_speed * DT;
        collided = expected > 0.0 && displacement < (expected * 0.2).max(2.0);

        state.time_since_collision = if collided {
            0.0
        } else {
            state.time_since_collision + DT
        };
        state.stuck_time = if displacement < 2.0 {
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

    let collision_touch = if state.time_since_collision < 0.6 {
        1.0
    } else {
        0.0
    };
    let jump_probe = if collided {
        if state.prev_jump != 0 { -1.0 } else { 1.0 }
    } else {
        0.0
    };
    let obs = [
        clamp(dx / WORLD_SIZE, -1.0, 1.0),
        clamp(dy / WORLD_SIZE, -1.0, 1.0),
        heading.sin(),
        heading.cos(),
        angle_error.sin(),
        angle_error.cos(),
        clamp(distance / WORLD_SIZE, 0.0, 1.0),
        clamp(state.stuck_time / 3.0, 0.0, 1.0),
        collision_touch,
        jump_probe,
    ];

    let mut steer_logits = [0.0; 7];
    let mut speed_logits = [0.0; 2];
    let mut h_out = [0.0; 96];
    let mut c_out = [0.0; 96];
    let inference_result = run_inference(
        state.net,
        obs.as_ptr(),
        state.h.as_ptr(),
        state.c.as_ptr(),
        steer_logits.as_mut_ptr(),
        speed_logits.as_mut_ptr(),
        h_out.as_mut_ptr(),
        c_out.as_mut_ptr(),
    );
    if inference_result != 0 {
        return inference_result;
    }
    state.h = h_out;
    state.c = c_out;

    let steer_bin = argmax(&steer_logits);
    let speed_bin = argmax(&speed_logits);
    let recovery = state.time_since_collision < 0.9
        || state.no_progress_time > 0.5
        || state.stuck_time > 0.5;
    let requested_direction = if recovery {
        heading + HEADING_OFFSETS[steer_bin]
    } else {
        target_angle + GOAL_OFFSETS[steer_bin]
    };
    let turn = clamp(normalize_angle(requested_direction - heading), -MAX_TURN, MAX_TURN);
    let direction = normalize_angle(heading + turn);
    let speed = if speed_bin == 1 { 100.0 } else { 50.0 };

    if !collided {
        state.jump_cooldown = 0;
    } else if state.prev_jump != 0 {
        state.jump_cooldown = JUMP_RETRY_COOLDOWN;
    } else if state.jump_cooldown > 0 {
        state.jump_cooldown -= 1;
    }
    let jump = if collided && state.prev_jump == 0 && state.jump_cooldown == 0 {
        1
    } else {
        0
    };

    *direction_out = direction;
    *speed_out = speed;
    *jump_out = jump;
    state.old_pos = [pos_x, pos_y];
    state.prev_dist = distance;
    state.prev_speed = speed;
    state.prev_jump = jump;
    state.first_step = false;
    0
}
