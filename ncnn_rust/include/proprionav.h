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
 *   abs_angle          absolute world-frame joystick/movement heading in radians
 *                      (y-up, 0 = +x/right, pi/2 = +y/up), maintained inside the
 *                      library and seeded with the bearing to the target on the
 *                      first call. Feed it straight to the joystick; do NOT also
 *                      accumulate turn_delta on top of it.
 *                      May be NULL to skip this output.
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
    int* jump,
    float* abs_angle
);

/* V5 feedback API: the game supplies the authoritative collision flag.
 * The macro output is the currently selected/active recovery macro id.
 * abs_angle has the same meaning as in nav_step and may be NULL.
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
    int* macro,
    float* abs_angle
);

/*
 * Runtime scale configuration. Must match the model's training environment:
 *   world_size     map extent in game units (e.g. 512 for the maze model)
 *   age_max_ms     max localisation age (obs normalisation)
 *   stale_ms       age at which position is treated as stale
 *   max_speed      max character speed (velocity normalisation)
 * Passing a value <= 0 keeps the current default for that field.
 * hidden size is detected automatically from the model param file.
 */
int nav_configure(
    void* nav,
    float world_size,
    float age_max_ms,
    float stale_ms,
    float max_speed
);

void nav_free(void* nav);

#ifdef __cplusplus
}
#endif

#endif
