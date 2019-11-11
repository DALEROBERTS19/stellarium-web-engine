/* Stellarium Web Engine - Copyright (c) 2018 - Noctua Software Ltd
 *
 * This program is licensed under the terms of the GNU AGPL v3, or
 * alternatively under a commercial licence.
 *
 * The terms of the AGPL v3 license can be found in the main directory of this
 * repository.
 */

#include <stdint.h>

/***** Labels manager *****************************************************/

void labels_reset(void);

/*
 * Function: labels_add
 * Render a label on screen.
 *
 * Parameters:
 *   text       - The text to render.
 *   pos        - 2D position of the text in screen in window (px).
 *   radius     - Radius of the point the label is linked to. Zero for
 *                independent label.
 *   size       - Height of the text in pixel.
 *   color      - Color of the text.
 *   angle      - Rotation angle (rad).
 *   align      - Union of <ALIGN_FLAGS> and LABEL_AROUND
 *   effects    - Union of <TEXT_EFFECT_FLAGS>.
 *   priority   - Priority used in case of positioning conflicts. Higher value
 *                means higher priority.
 *   oid        - Optional unique id for the label.
 */
void labels_add(const char *text, const double pos[2], double radius,
                double size, const double color[4], double angle,
                int align, int effects, double priority, uint64_t oid);

/*
 * Function: labels_add
 * Render a label on screen.
 *
 * Parameters:
 *   text       - The text to render.
 *   pos        - 3D position of the text in given frame.
 *   at_inf     - true if the objec is at infinity (pos is normalized).
 *   frame      - One of FRAME_XXX.
 *   radius     - Radius of the point the label is linked to. Zero for
 *                independent label.
 *   size       - Height of the text in pixel.
 *   color      - Color of the text.
 *   angle      - Rotation angle (rad).
 *   align      - Union of <ALIGN_FLAGS> and LABEL_AROUND
 *   effects    - Union of <TEXT_EFFECT_FLAGS>.
 *   priority   - Priority used in case of positioning conflicts. Higher value
 *                means higher priority.
 *   oid        - Optional unique id for the label.
 */
void labels_add_3d(const char *text, int frame, const double pos[3],
                   bool at_inf, double radius, double size,
                   const double color[4], double angle, int align,
                   int effects, double priority, uint64_t oid);
