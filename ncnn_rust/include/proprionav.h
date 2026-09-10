#ifndef PROPRIONAV_H
#define PROPRIONAV_H

#ifdef __cplusplus
extern "C" {
#endif

/* Load one model and create an isolated stateful navigation session. */
void* nav_init(const char* param_path, const char* bin_path);

/*
 * Inputs:
 *   pos_x, pos_y       current character position
 *   target_x, target_y current target or waypoint position
 *   position_age_ms    age of the supplied position observation in ms
 *
 * Outputs:
 *   turn_delta         relative turn from the previous movement direction
 *                      in radians [-pi/4, pi/4]
 *   speed              freshness-limited recommended speed [0, 100]
 *   jump               0 or 1
 *
 * Returns 0 on success, a negative error code on failure.
 */
int nav_step(
    void* nav,
    float pos_x,
    float pos_y,
    float target_x,
    float target_y,
    float position_age_ms,
    float* turn_delta,
    float* speed,
    int* jump
);

/* V5 feedback API: the game supplies the authoritative collision flag.
 * The macro output is the currently selected/active recovery macro id.
 */
int nav_step_feedback(
    void* nav,
    float pos_x,
    float pos_y,
    float target_x,
    float target_y,
    float position_age_ms,
    int collided,
    float* turn_delta,
    float* speed,
    int* jump,
    int* macro
);

void nav_free(void* nav);

#ifdef __cplusplus
}
#endif

#endif
