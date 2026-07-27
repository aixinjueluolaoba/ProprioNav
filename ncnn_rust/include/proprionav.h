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
 *   target_x, target_y current target position
 *   heading            current character heading in radians
 *
 * Outputs:
 *   direction          movement direction in radians [-pi, pi]
 *   speed              50 or 100
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
    float heading,
    float* direction,
    float* speed,
    int* jump
);

void nav_free(void* nav);

#ifdef __cplusplus
}
#endif

#endif
