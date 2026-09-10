#![allow(non_camel_case_types)]

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
const JUMP_RETRY_COOLDOWN: i32 = 8;
const RECOVERY_COMMIT_STEPS: i32 = 4;
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
/// Inputs: x[13], h_in[96], c_in[96]
/// Outputs: steer_logits[7], speed_logits[2], h_out[96], c_out[96]
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
) -> i32 {
    if net.is_null() || x.is_null() || h_in.is_null() || c_in.is_null() {
        return -1;
    }

    let extractor = ncnn_extractor_create(net);
    if extractor.is_null() {
        return -2;
    }
    let mat_x = ncnn_mat_create_external_1d(OBS_DIM, x as *mut c_void, std::ptr::null_mut());
    let mat_h = ncnn_mat_create_external_1d(96, h_in as *mut c_void, std::ptr::null_mut());
    let mat_c = ncnn_mat_create_external_1d(96, c_in as *mut c_void, std::ptr::null_mut());
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
            std::ptr::copy_nonoverlapping(ncnn_mat_get_data(mat_out2), h_out, 96);
        }
        if !c_out.is_null() {
            std::ptr::copy_nonoverlapping(ncnn_mat_get_data(mat_out3), c_out, 96);
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
        std::ptr::null_mut(),
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
        net, x, h_in, c_in, steer_logits, speed_logits, h_out, c_out, macro_logits,
    )
}

fn freshness_scale(age_ms: f32) -> f32 {
    if age_ms <= 200.0 {
        1.0
    } else if age_ms <= MAX_POSITION_AGE_MS {
        1.0 - 0.5 * ((age_ms - 200.0) / 300.0)
    } else if age_ms <= STALE_POSITION_AGE_MS {
        0.5 * ((STALE_POSITION_AGE_MS - age_ms) / 500.0)
    } else {
        0.0
    }
}

struct NavState {
    net: ncnn_net_t,
    h: [f32; 96],
    c: [f32; 96],
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
    jump_cooldown: i32,
    recovery_commit_left: i32,
    recovery_phase: i32,
    recovery_bin: usize,
    macro_left: i32,
    macro_target_heading: f32,
    active_macro: i32,
}

impl NavState {
    fn new(net: ncnn_net_t) -> Self {
        Self {
            net,
            h: [0.0; 96],
            c: [0.0; 96],
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
            jump_cooldown: 0,
            recovery_commit_left: 0,
            recovery_phase: 0,
            recovery_bin: 6,
            macro_left: 0,
            macro_target_heading: 0.0,
            active_macro: 0,
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

/// V4 navigation API without a heading input.
///
/// Inputs: current position, fixed target position, position age in ms.
/// Outputs: relative turn delta in radians, freshness-limited speed, jump.
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
        collision_measurement_valid = sample_time_valid && age_ms <= STALE_POSITION_AGE_MS;
        if collided_override.is_none() {
            let expected = state.prev_speed * sample_dt.max(MIN_SAMPLE_DT);
            collided = expected > 0.0
                && collision_measurement_valid
                && displacement < (expected * 0.2).max(2.0);
        } else {
            collision_measurement_valid = age_ms <= STALE_POSITION_AGE_MS;
        }
    }

    let sample_time_valid = (MIN_SAMPLE_DT..=MAX_SAMPLE_DT).contains(&sample_dt);
    let sample_speed = displacement / sample_dt.max(MIN_SAMPLE_DT);
    let velocity_valid = sample_time_valid
        && displacement >= 2.0
        && sample_speed <= MAX_REASONABLE_SPEED;
    if velocity_valid {
        let mut sample_velocity = [
            (pos_x - state.old_pos[0]) / sample_dt,
            (pos_y - state.old_pos[1]) / sample_dt,
        ];
        let sample_norm = (sample_velocity[0] * sample_velocity[0]
            + sample_velocity[1] * sample_velocity[1])
            .sqrt();
        if sample_norm > MAX_REASONABLE_SPEED {
            let scale = MAX_REASONABLE_SPEED / sample_norm;
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
        let prediction_age = age_ms.min(MAX_POSITION_AGE_MS) / 1000.0;
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
    }

    let prediction_age = age_ms.min(MAX_POSITION_AGE_MS) / 1000.0;
    let predicted_x = clamp(pos_x + state.velocity[0] * prediction_age, -1100.0, 1100.0);
    let predicted_y = clamp(pos_y + state.velocity[1] * prediction_age, -1100.0, 1100.0);
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
        clamp(dx / WORLD_SIZE, -1.0, 1.0),
        clamp(dy / WORLD_SIZE, -1.0, 1.0),
        angle_error.sin(),
        angle_error.cos(),
        clamp(state.velocity[0] / MAX_REASONABLE_SPEED, -1.0, 1.0),
        clamp(state.velocity[1] / MAX_REASONABLE_SPEED, -1.0, 1.0),
        clamp(distance / WORLD_SIZE, 0.0, 1.0),
        clamp(state.stuck_time / 3.0, 0.0, 1.0),
        collision_touch,
        jump_probe,
        clamp(age_ms / MAX_POSITION_AGE_MS, 0.0, 1.0),
        state.heading_confidence,
        clamp(state.last_turn_delta / MAX_TURN, -1.0, 1.0),
    ];

    let mut steer_logits = [0.0; 7];
    let mut speed_logits = [0.0; 2];
    let mut macro_logits = [0.0; 8];
    let mut h_out = [0.0; 96];
    let mut c_out = [0.0; 96];
    let inference_result = run_inference_macro(
        state.net,
        obs.as_ptr(),
        state.h.as_ptr(),
        state.c.as_ptr(),
        steer_logits.as_mut_ptr(),
        speed_logits.as_mut_ptr(),
        h_out.as_mut_ptr(),
        c_out.as_mut_ptr(),
        if macro_enabled { macro_logits.as_mut_ptr() } else { std::ptr::null_mut() },
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
    let base_speed = if speed_bin == 1 { 100.0 } else { 50.0 };
    let speed = base_speed * freshness_scale(age_ms);

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

    *turn_delta_out = turn_delta;
    *speed_out = speed;
    *jump_out = jump;
    if !macro_out.is_null() {
        *macro_out = state.active_macro;
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
