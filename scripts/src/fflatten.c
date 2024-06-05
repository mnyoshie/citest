/* fflaten.c - fast flatten
 *
 * Copyright (C) 2024 Minato Yoshie & Al-buharie Amjari
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation; either version 2 of the License, or (at your
 * option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * for more details.
 * 
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, 51
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <png.h>


#define FFMAX(a, b, c) ( a > b ? (a > c ? a : c): ((b > c) ? b : c))
#define FFMIN(a, b, c) ( a < b ? (a < c ? a : c): ((b < c) ? b : c))

#define FFMAXP(a, b, c) ( *a > *b ? (*a > *c ? a : c): ((*b > *c) ? b : c))
#define FFMIDP(a, b, c) ((*a > *b && *b > *c) ? b : ((*b > *c && *c > *a) ? c : a))
#define FFMINP(a, b, c) ( *a < *b ? (*a < *c ? a : c): ((*b < *c) ? b : c))

//#define FFDEBUB 1

/* for some reason, commenting this results in faster execution on my 
 * ARM machine. If things go hay wire, comment this:
 */
//#define FFLATTEN_ENABLE_INTRINSICS

#if (defined(__ARM_NEON) || defined(__ARM_NEON__)) && \
     defined(FFLATTEN_ENABLE_INTRINSICS)
#define FFLATTEN_INTRINSICS_USED
/* load nyan intrinsics */
#  include <arm_neon.h>
_Static_assert(sizeof(float32_t) == 4, "FLOAT must be 32 bits. Fix me\n");

#elif (defined(__i386__) || defined(__x86_64__)) && \
       defined(FFLATTEN_ENABLE_INTRINSICS)
#define FFLATTEN_INTRINSICS_USED
/* load nyan to sse intrinsics */
#  include "NEON_2_SSE.h"
_Static_assert(sizeof(float32_t) == 4, "FLOAT must be 32 bits. Fix me\n");
#endif /* (defined(__ARM_NEON) || defined(__ARM_NEON__)) &&
           defined(FFLATTEN_ENABLE_INTRINSICS) */


#if !defined(FFLATTEN_INTRINSICS_USED)
#warning "using low quality intrinsics"
// for portability, we unroll nyan intrinsics

typedef float float32_t;
_Static_assert(sizeof(float32_t) == 4, "FLOAT must be 32 bits. Fix me\n");

typedef struct float32x4_t float32x4_t;
struct float32x4_t {
  float32_t x, y, z, w;
};

typedef struct uint32x4_t uint32x4_t;
struct uint32x4_t {
  uint32_t x, y, z, w;
};

// XXX CONVERSION
static inline uint32x4_t vcvtq_u32_f32(float32x4_t a) {
  return (uint32x4_t){
    (uint32_t) a.x, (uint32_t) a.y,
    (uint32_t) a.z, (uint32_t) a.w
  };
}

static inline uint32_t vgetq_lane_u32(uint32x4_t a, int b) {
  switch (b) {
  case 0: return a.x; case 1: return a.y;
  case 2: return a.z; case 3: return a.w;
  default:
  abort();
  }
}

static inline float32_t vgetq_lane_f32(float32x4_t a, int b) {
  switch (b) {
  case 0: return a.x; case 1: return a.y;
  case 2: return a.z; case 3: return a.w;
  default:
  abort();
  }
}

// XXX COMPARISON
static inline uint32x4_t vcgtq_f32(float32x4_t a, float32x4_t b) {
  return (uint32x4_t){a.x > b.x, a.y > b.y, a.z > b.z, a.w > b.w};
}

static inline uint32x4_t vcltq_f32(float32x4_t a, float32x4_t b) {
  return (uint32x4_t){a.x < b.x, a.y < b.y, a.z < b.z, a.w < b.w};
}

static inline uint32x4_t vcgeq_f32(float32x4_t a, float32x4_t b) {
  return (uint32x4_t){a.x >= b.x, a.y >= b.y, a.z >= b.z, a.w >= b.w};
}

static inline uint32x4_t vcleq_f32(float32x4_t a, float32x4_t b) {
  return (uint32x4_t){a.x <= b.x, a.y <= b.y, a.z <= b.z, a.w <= b.w};
}

static inline uint32x4_t vceqq_f32(float32x4_t a, float32x4_t b) {
  return (uint32x4_t){a.x == b.x, a.y == b.y, a.z == b.z, a.w == b.w};
}

static inline float32x4_t vminq_f32(float32x4_t a, float32x4_t b) {
  return (float32x4_t){
    a.x < b.x ? a.x : b.x,
    a.y < b.y ? a.y : b.y,
    a.z < b.z ? a.z : b.z,
    a.w < b.w ? a.w : b.w
  };
}

static inline float32x4_t vbslq_f32(uint32x4_t mask, float32x4_t a,
                                    float32x4_t b) {
  float x = (mask.x ? a.x : b.x), y = (mask.y ? a.y : b.y);
  float z = (mask.z ? a.z : b.z), w = (mask.w ? a.w : b.w);
  return (float32x4_t){x, y, z, w};
}

// XXX LOAD
static inline float32x4_t vld1q_f32(float *a) {
  return (float32x4_t){a[0], a[1], a[2], a[3]};
}

// XXX STORE
static inline void vst1q_f32(float *a, float32x4_t b) {
  a[0] = b.x; a[1] = b.y; a[2] = b.z; a[3] = b.w;
}

// XXX ADD
static inline float32x4_t vaddq_f32(float32x4_t a, float32x4_t b) {
  return (float32x4_t){a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

static inline float32x4_t vaddq_n_f32(float32x4_t a, float32_t b) {
  return (float32x4_t){a.x + b, a.y + b, a.z + b, a.w + b};
}

static inline float32_t vaddvq_f32(float32x4_t a) {
  return a.x + a.y + a.z + a.w;
}
// XXX SUB
static inline float32x4_t vsubq_f32(float32x4_t a, float32x4_t b) {
  return (float32x4_t){a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}

static inline float32x4_t vsubq_n_f32(float32x4_t a, float32_t b) {
  return (float32x4_t){a.x - b, a.y - b, a.z - b, a.w - b};
}

// XXX MULTIPLICATION
static inline float32x4_t vmulq_f32(float32x4_t a, float32x4_t b) {
  return (float32x4_t){a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
}

static inline float32x4_t vmulq_n_f32(float32x4_t a, float b) {
  return (float32x4_t){a.x * b, a.y * b, a.z * b, a.w * b};
}

// XXX DIVISION

static inline float32_t ffrec(float32_t a) {
  return 1.0 / a;
}

static inline float32x4_t vrecpeq_f32(float32x4_t a) {
  return (float32x4_t){
    ffrec(a.x),
    ffrec(a.y),
    ffrec(a.z),
    ffrec(a.w)
  };
}

// XXX MISCS
static inline float32_t ffabs(float32_t a) {
  uint32_t mask = ~(1 << 31);
  uint32_t *val = (void *)&a;
  float32_t ret = *(float32_t *)&(uint32_t){*val & mask};
  return ret;
}

static inline float32x4_t vabsq_f32(float32x4_t a) {
  return (float32x4_t){
    ffabs(a.x),
    ffabs(a.y),
    ffabs(a.z),
    ffabs(a.w)
  };
}

static inline float32x4_t vsqrtq_f32(float32x4_t a) {
  return (float32x4_t){
    sqrtf(a.x),
    sqrtf(a.y),
    sqrtf(a.z),
    sqrtf(a.w)
  };
}
#endif /* !defined(FFLATTEN_INTRINSICS_USED) */

/* not neon defined function. prefixed with ff, incase future neon
 * incorporates such intrinsics
 */

static inline float32x4_t ffvpowq_f32(float32x4_t a, float32x4_t b) {
  return (float32x4_t){
    powf(a.x, b.x),
    powf(a.y, b.y),
    powf(a.z, b.z),
    powf(a.w, b.w)
  };
}

#define FFSCREEN(base, top) vsubq_f32(vaddq_f32(base, top), vmulq_f32(base, top))
#define FFMULTIPLY(base, top) vmulq_f32(base, top)


/* primitive W3C blend modes
 * see https://www.w3.org/TR/compositing-1
 *
 * This implementation does not guarantee
 * conformance.
 */
#define BLEND_BASE         ((char)'0')
#define BLEND_TOP          ((char)'1')
#define BLEND_NORMAL       ((char)' ')
#define BLEND_ADDITION     ((char)'+')
#define BLEND_COLOR        ((char)'@')
#define BLEND_COLOR_DODGE  ((char)'\\')
#define BLEND_DARKEN       ((char)'D')
#define BLEND_DIFFERENCE   ((char)'d')
#define BLEND_DIVIDE       ((char)'/')
#define BLEND_GAMMA_LIGHT  ((char)'g')
#define BLEND_GAMMA_DARK   ((char)'k')
#define BLEND_HUE          ((char)'h')
#define BLEND_HARD_LIGHT   ((char)'H')
#define BLEND_LIGHTEN      ((char)'L')
#define BLEND_LUMINOSITY   ((char)'u')
#define BLEND_MULTIPLY     ((char)'*')
#define BLEND_OVERLAY      ((char)'_')
#define BLEND_SATURATION   ((char)'s')
#define BLEND_SCREEN       ((char)'c')
#define BLEND_SOFT_LIGHT   ((char)'S')


#define PRINT_BLEND_OP(x) printf("  '%c'         %s\n", BLEND_##x, #x) 

/* base class for all blend modes */
#define DEFINE_BLEND_FUNC(name)                                                  \
  static void blend_##name(imgf32_t *restrict base, imgf32_t *restrict top) {    \
    for (uint32_t y = 0; y < base->height; y++) {                                \
      float32_t *brow = base->rows[y];                                           \
      float32_t *trow = top->rows[y];                                            \
      for (uint32_t x = 0; x < base->width; x++) {                               \
        float32_t *(bpx) = (void *)(brow + x * 4);                               \
        float32_t *(tpx) = (void *)(trow + x * 4);                               \
        bpx[3] *= base->opacity;                                                 \
        tpx[3] *= top->opacity;                                                  \
        BLEND_BODY_##name;                                                       \
      }                                                                          \
    }                                                                            \
    base->opacity = 1.0;                                                         \
  }

#define DEFINE_BLEND_CASE(name, base, top)                     \
  case BLEND_##name: blend_##name(base, top); break

/* Apply the blend in place
 * 
 * Cs = (1 - αb) x Cs + αb x B(Cb, Cs)
 * Composite
 * 
 * Co = αs x Fa x Cs + αb x Fb x Cb
 */

#define FFCS(B) vaddq_f32(vmulq_n_f32(freg_top, 1.0-bpx[3]), vmulq_n_f32(B, bpx[3]))
/* Fa and Fb, porter-duff values */
#define FFCO(Fa, Fb, B)                       \
  vaddq_f32(vmulq_n_f32(FFCS(B), Fa* tpx[3]), \
            vmulq_n_f32(freg_base, Fb* bpx[3]))

/* When writing a blend body, only intrinsics are allowed
 * and branching is prohibited. 
 */

// XXX SEPARABLE BLEND MODES

#define BLEND_BODY_BASE                                        \
  do {                                                         \
  } while (0)

#define BLEND_BODY_TOP                                         \
  do {                                                         \
    float32x4_t freg_top = vld1q_f32(tpx);                     \
    vst1q_f32(bpx, freg_top);                                  \
  } while (0)

#define BLEND_BODY_NORMAL                                      \
  do {                                                         \
    float32x4_t freg_base = vld1q_f32(bpx);                    \
    float32x4_t freg_top = vld1q_f32(tpx);                     \
    freg_base = FFCO(1.0, (1.0 - tpx[3]), freg_top);           \
    vst1q_f32(bpx, freg_base);                                 \
    bpx[3] = 1.0; \
  } while (0)

#define BLEND_BODY_ADDITION                                    \
  do {                                                         \
    const float32x4_t ffour_ones = {1.0, 1.0, 1.0, 1.0};       \
    float32x4_t freg_base = vld1q_f32(bpx);                    \
    float32x4_t freg_top = vld1q_f32(tpx);                     \
    freg_base = vaddq_f32(freg_base, freg_top);                \
    uint32x4_t mask = vcgtq_f32(freg_base, ffour_ones);        \
    freg_base = vbslq_f32(mask, ffour_ones, freg_base);        \
    vst1q_f32(bpx, freg_base);                                 \
  } while (0)

#define BLEND_BODY_COLOR_DODGE                                 \
  do {                                                         \
    const float32x4_t ffour_zeros = {0.0, 0.0, 0.0, 0.0};      \
    const float32x4_t ffour_ones = {1.0, 1.0, 1.0, 1.0};       \
    const float32x4_t ffour_czeros = {0.001, 0.001, 0.02, 0.001}; \
    const float32x4_t ffour_cones = {0.99, 0.99, 0.97, 0.98};  \
    float32x4_t freg_base = vld1q_f32(bpx);                    \
    float32x4_t freg_top = vld1q_f32(tpx);                     \
    uint32x4_t freg_eq0 = vcltq_f32(freg_base, ffour_czeros);  \
    uint32x4_t freg_eq1 = vcgtq_f32(freg_top, ffour_cones);    \
    float32x4_t freg_rec =                                     \
      vrecpeq_f32(vsubq_f32(ffour_ones, freg_top));            \
    float32x4_t freg_min =                                     \
      vminq_f32(ffour_ones, vmulq_f32(freg_base, freg_rec));   \
    freg_base = FFCO(1.0, 1.0-tpx[3], vbslq_f32(freg_eq0,      \
      ffour_zeros, vbslq_f32(freg_eq1, ffour_ones, freg_min)));\
    vst1q_f32(bpx, freg_base);                                 \
    bpx[3] = 1.0; \
  } while (0)

#define BLEND_BODY_DIFFERENCE                                  \
  do {                                                         \
    const float32x4_t ffour_ones = {1.0, 1.0, 1.0, 1.0};       \
    float32x4_t freg_base = vld1q_f32(bpx);                    \
    float32x4_t freg_top = vld1q_f32(tpx);                     \
    uint32x4_t mask = vcgtq_f32(freg_base, freg_top);          \
    float32x4_t freg_max =                                     \
      vbslq_f32(mask, freg_base, freg_top);                    \
    mask = vcltq_f32(freg_base, freg_top);                     \
    float32x4_t freg_min =                                     \
      vbslq_f32(mask, freg_base, freg_top);                    \
    float32x4_t freg_dif = vsubq_f32(freg_max, freg_min);      \
    float32x4_t freg_abs = vabsq_f32(freg_dif);                \
    mask = vcgtq_f32(freg_abs, ffour_ones);                    \
    freg_base = FFCO(1.0, 1.0 - tpx[3],                        \
      vbslq_f32(mask, ffour_ones, freg_abs));                  \
    vst1q_f32(bpx, freg_base);                                 \
    bpx[3] = 1.0; \
  } while (0)

#define BLEND_BODY_SCREEN                                      \
  do {                                                         \
    const float32x4_t ffour_ones = {1.0, 1.0, 1.0, 1.0};       \
    float32x4_t freg_base = vld1q_f32(bpx);                    \
    float32x4_t freg_top = vld1q_f32(tpx);                     \
    float32x4_t freg_screen = FFSCREEN(freg_base, freg_top);   \
    freg_base = FFCO(1.0, 1.0-tpx[3], freg_screen);            \
    uint32x4_t mask = vcgtq_f32(freg_base, ffour_ones);        \
    freg_base = vbslq_f32(mask, ffour_ones, freg_base);        \
    vst1q_f32(bpx, freg_base);                                 \
    bpx[3] = 1.0;                                              \
  } while (0)

#define BLEND_BODY_HARD_LIGHT                                  \
  do {                                                         \
    const float32x4_t ffour_ones = {1.0, 1.0, 1.0, 1.0};       \
    const float32x4_t ffour_halfs = {0.5, 0.5, 0.5, 0.5};      \
    float32x4_t freg_base = vld1q_f32(bpx);                    \
    float32x4_t freg_top = vld1q_f32(tpx);                     \
    float32x4_t freg_mult = FFMULTIPLY(freg_base,              \
      vmulq_n_f32(freg_top, 2.0));                             \
    float32x4_t freg_screen = FFSCREEN(freg_base,              \
      vsubq_f32(vmulq_n_f32(freg_top, 2.0), ffour_ones));      \
    uint32x4_t mask = vcltq_f32(freg_top, ffour_halfs);        \
    freg_base = FFCO(1.0, 1.0-tpx[3],                          \
      vbslq_f32(mask, freg_mult, freg_screen));                \
    vst1q_f32(bpx, freg_base);                                 \
    bpx[3] = 1.0;                                              \
  } while (0)


#define BLEND_BODY_OVERLAY                                     \
  do {                                                         \
    const float32x4_t ffour_ones = {1.0, 1.0, 1.0, 1.0};       \
    const float32x4_t ffour_halfs = {0.5, 0.5, 0.5, 0.5};      \
    float32x4_t freg_base = vld1q_f32(bpx);                    \
    float32x4_t freg_top = vld1q_f32(tpx);                     \
    float32x4_t freg_mult = FFMULTIPLY(freg_top,               \
      vmulq_n_f32(freg_base, 2.0));                            \
    float32x4_t freg_screen = FFSCREEN(freg_top,               \
      vsubq_f32(vmulq_n_f32(freg_base, 2.0), ffour_ones));     \
    uint32x4_t mask = vcltq_f32(freg_base, ffour_halfs);       \
    freg_base = FFCO(1.0, 1.0-tpx[3],                          \
      vbslq_f32(mask, freg_mult, freg_screen));                \
    vst1q_f32(bpx, freg_base);                                 \
    bpx[3] = 1.0;                                              \
  } while (0)

#define BLEND_BODY_SOFT_LIGHT                                                 \
  do {                                                                        \
    const float32x4_t ffour_fours = {4.0, 4.0, 4.0, 4.0};                     \
    const float32x4_t ffour_twelves = {12.0, 12.0, 12.0, 12.0};               \
    const float32x4_t ffour_ones = {1.0, 1.0, 1.0, 1.0};                      \
    const float32x4_t ffour_halfs = {0.5, 0.5, 0.5, 0.5};                     \
    const float32x4_t ffour_hhalfs = {0.25, 0.25, 0.25, 0.25};                \
    float32x4_t freg_base = vld1q_f32(bpx);                                   \
    float32x4_t freg_top = vld1q_f32(tpx);                                    \
    float32x4_t freg_d025 = vmulq_f32(                                        \
        vaddq_f32(                                                            \
            vmulq_f32(vsubq_f32(vmulq_n_f32(freg_base, 16.0), ffour_twelves), \
                      freg_base),                                             \
            ffour_fours),                                                     \
        freg_base);                                                           \
    float32x4_t freg_e025 = vsqrtq_f32(freg_base);                            \
    float32x4_t freg_d050 = vsubq_f32(                                        \
        freg_base,                                                            \
        vmulq_f32(vsubq_f32(ffour_ones, vmulq_n_f32(freg_top, 2.0)),          \
                  vmulq_f32(freg_base, vsubq_f32(ffour_ones, freg_base))));   \
    uint32x4_t mask = vcleq_f32(freg_base, ffour_hhalfs);                     \
    float32x4_t freg_e050 = vaddq_f32(                                        \
        freg_base,                                                            \
        vmulq_f32(                                                            \
            vsubq_f32(vmulq_n_f32(freg_top, 2.0), ffour_ones),                \
            vsubq_f32(vbslq_f32(mask, freg_d025, freg_e025), freg_base)));    \
    mask = vcleq_f32(freg_top, ffour_halfs);                                  \
    freg_base =                                                               \
        FFCO(1.0, 1.0 - tpx[3], vbslq_f32(mask, freg_d050, freg_e050));       \
    vst1q_f32(bpx, freg_base);                                                \
    bpx[3] = 1.0;                                                             \
  } while (0)


#define BLEND_BODY_DARKEN                                      \
  do {                                                         \
    float32x4_t freg_base = vld1q_f32(bpx);                    \
    float32x4_t freg_top = vld1q_f32(tpx);                     \
    uint32x4_t mask = vcltq_f32(freg_top, freg_base);          \
    freg_base = FFCO(1.0, 1.0-tpx[3],                          \
      vbslq_f32(mask, freg_top, freg_base));                   \
    vst1q_f32(bpx, freg_base);                                 \
    bpx[3] = 1.0;                                              \
  } while (0)


#define BLEND_BODY_GAMMA_LIGHT                                 \
  do {                                                         \
    float32x4_t freg_base = vld1q_f32(bpx);                    \
    float32x4_t freg_top = vld1q_f32(tpx);                     \
    freg_base = FFCO(1.0, 1.0-tpx[3],                          \
      ffvpowq_f32(freg_base, freg_top));                       \
    vst1q_f32(bpx, freg_base);                                 \
  } while (0)

#define BLEND_BODY_GAMMA_DARK                                  \
  do {                                                         \
    const float32x4_t ffour_zeros = {0.0, 0.0, 0.0, 0.0};      \
    float32x4_t freg_base = vld1q_f32(bpx);                    \
    float32x4_t freg_top = vld1q_f32(tpx);                     \
    uint32x4_t mask = vceqq_f32(freg_top, ffour_zeros);        \
    float32x4_t freg_gdark = vbslq_f32(mask, ffour_zeros,      \
      ffvpowq_f32(freg_base, vrecpeq_f32(freg_top)));          \
    freg_base = FFCO(1.0, 1.0-tpx[3], freg_gdark);             \
    vst1q_f32(bpx, freg_base);                                 \
    bpx[3] = 1.0;                                              \
  } while (0)

#define BLEND_BODY_LIGHTEN                                     \
  do {                                                         \
    float32x4_t freg_base = vld1q_f32(bpx);                    \
    float32x4_t freg_top = vld1q_f32(tpx);                     \
    uint32x4_t mask = vcgtq_f32(freg_top, freg_base);          \
    freg_base = FFCO(1.0, 1.0-tpx[3],                          \
      vbslq_f32(mask, freg_top, freg_base));                   \
    vst1q_f32(bpx, freg_base);                                 \
    bpx[3] = 1.0;                                              \
  } while (0)

#define BLEND_BODY_DIVIDE                                      \
  do {                                                         \
    const float32x4_t ffour_ones = {1.0, 1.0, 1.0, 1.0};       \
    float32x4_t freg_base = vld1q_f32(bpx);                    \
    float32x4_t freg_top = vld1q_f32(tpx);                     \
    float32x4_t freg_rec = vrecpeq_f32(freg_top);              \
    float32x4_t freg_mul = vmulq_f32(freg_base, freg_rec);     \
    uint32x4_t mask = vcgtq_f32(freg_mul, ffour_ones);         \
    freg_base = FFCO(1.0, 1.0-tpx[3],                          \
      vbslq_f32(mask, ffour_ones, freg_mul));                  \
    vst1q_f32(bpx, freg_base);                                 \
  } while (0)

#define BLEND_BODY_MULTIPLY                                    \
  do {                                                         \
    float32x4_t freg_base = vld1q_f32(bpx);                    \
    float32x4_t freg_top = vld1q_f32(tpx);                     \
    float32x4_t freg_tmp = FFMULTIPLY(freg_base, freg_top);    \
    freg_base = FFCO(1.0, (1.0 - tpx[3]), freg_tmp);           \
    vst1q_f32(bpx, freg_base);                                 \
    bpx[3] = 1.0;\
  } while (0)


/* The nonseparable blend mode formulas make use of several auxiliary functions:
 * 
 *     Lum(C) = 0.3 x Cred + 0.59 x Cgreen + 0.11 x Cblue
 *     
 *     ClipColor(C)
 *         L = Lum(C)
 *         n = min(Cred, Cgreen, Cblue)
 *         x = max(Cred, Cgreen, Cblue)
 *         if(n < 0)
 *             C = L + (((C - L) * L) / (L - n))
 *                       
 *         if(x > 1)
 *             C = L + (((C - L) * (1 - L)) / (x - L))
 *         
 *         return C
 *     
 *     SetLum(C, l)
 *         d = l - Lum(C)
 *         Cred = Cred + d
 *         Cgreen = Cgreen + d
 *         Cblue = Cblue + d
 *         return ClipColor(C)
 *         
 *     Sat(C) = max(Cred, Cgreen, Cblue) - min(Cred, Cgreen, Cblue)
 *       
 * The subscripts min, mid, and max in the next function refer to the color
 * components having the minimum, middle, and maximum values upon entry to the function.
 * 
 *     SetSat(C, s)
 *         if(Cmax > Cmin)
 *             Cmid = (((Cmid - Cmin) x s) / (Cmax - Cmin))
 *             Cmax = s
 *         else
 *             Cmid = Cmax = 0
 *         Cmin = 0
 */

static inline float32_t rgb2lum(float32x4_t rgb) {
 
  const float32x4_t wcc= {0.3, 0.59, 0.11, 0.0}; 
  return vaddvq_f32(vmulq_f32(rgb, wcc));
}

static inline float32_t rgb2sat(float32x4_t rgb) {
  float32_t r = vgetq_lane_f32(rgb, 0);
  float32_t g = vgetq_lane_f32(rgb, 1);
  float32_t b = vgetq_lane_f32(rgb, 2);

  return FFMAX(r, g, b) - FFMIN(r, g,  b);
}

static inline float32x4_t clip_color(float32x4_t rgb) {
  float32_t r = vgetq_lane_f32(rgb, 0);
  float32_t g = vgetq_lane_f32(rgb, 1);
  float32_t b = vgetq_lane_f32(rgb, 2);
  float32_t L = rgb2lum(rgb);
  float32_t n = FFMIN(r, g, b);
  float32_t x = FFMAX(r, g, b);

  if (n < 0.0) {
    r = L + (((r - L) * L) / (L - n));
    g = L + (((g - L) * L) / (L - n));
    b = L + (((b - L) * L) / (L - n));
  }

  if (x > 1.0) {
    r = L + (((r - L) * (1 - L)) / (x - L));
    g = L + (((g - L) * (1 - L)) / (x - L));
    b = L + (((b - L) * (1 - L)) / (x - L));
  }

  return (float32x4_t){r, g, b, vgetq_lane_f32(rgb, 3)};
}

static inline float32x4_t set_lum(float32x4_t rgb, float32_t l) {
  float32_t r = vgetq_lane_f32(rgb, 0);
  float32_t g = vgetq_lane_f32(rgb, 1);
  float32_t b = vgetq_lane_f32(rgb, 2);

  float32_t d = l - rgb2lum(rgb);
  r += d;
  g += d;
  b += d;
  return clip_color((float32x4_t){r, g, b, vgetq_lane_f32(rgb, 3)});
}

static inline float32x4_t set_sat(float32x4_t rgb, float32_t s) {
  float32_t r = vgetq_lane_f32(rgb, 0);
  float32_t g = vgetq_lane_f32(rgb, 1);
  float32_t b = vgetq_lane_f32(rgb, 2);
  float32_t *max = FFMAXP(&r, &g, &b);
  float32_t *mid = FFMIDP(&r, &g, &b);
  float32_t *min = FFMINP(&r, &g, &b);

  if (*max > *min) {
    *mid = (((*mid - *min) * s) / (*max - *min));
    *max = s;
  } else {
    *mid = 0.0;
    *max = 0.0;
  }

  *min = 0.0;

  return (float32x4_t){r, g, b, vgetq_lane_f32(rgb, 3)};
}

// XXX NON-SEPARABLE BLEND MODES

#define BLEND_BODY_COLOR                                                \
  do {                                                                  \
    float32x4_t freg_base = vld1q_f32(bpx);                             \
    float32x4_t freg_top = vld1q_f32(tpx);                              \
    freg_base =                                                         \
        FFCO(1.0, 1.0 - tpx[3], set_lum(freg_top, rgb2lum(freg_base))); \
    vst1q_f32(bpx, freg_base);                                          \
    bpx[3] = 1.0;                                                       \
  } while (0)

#define BLEND_BODY_HUE                                                       \
  do {                                                                       \
    float32x4_t freg_base = vld1q_f32(bpx);                                  \
    float32x4_t freg_top = vld1q_f32(tpx);                                   \
    freg_base = FFCO(                                                        \
        1.0, 1.0 - tpx[3],                                                   \
        set_lum(set_sat(freg_top, rgb2sat(freg_base)), rgb2lum(freg_base))); \
    vst1q_f32(bpx, freg_base);                                               \
    bpx[3] = 1.0;                                                            \
  } while (0)

#define BLEND_BODY_SATURATION                                                \
  do {                                                                       \
    float32x4_t freg_base = vld1q_f32(bpx);                                  \
    float32x4_t freg_top = vld1q_f32(tpx);                                   \
    freg_base = vmulq_n_f32(freg_base, base->opacity);                       \
    freg_top = vmulq_n_f32(freg_top, top->opacity);                          \
    freg_base = FFCO(                                                        \
        1.0, 1.0 - tpx[3],                                                   \
        set_lum(set_sat(freg_base, rgb2sat(freg_top)), rgb2lum(freg_base))); \
    vst1q_f32(bpx, freg_base);                                               \
    bpx[3] = 1.0;                                                            \
  } while (0)

#define BLEND_BODY_LUMINOSITY                                           \
  do {                                                                  \
    float32x4_t freg_base = vld1q_f32(bpx);                             \
    float32x4_t freg_top = vld1q_f32(tpx);                              \
    freg_base = vmulq_n_f32(freg_base, base->opacity);                  \
    freg_top = vmulq_n_f32(freg_top, top->opacity);                     \
    freg_base =                                                         \
        FFCO(1.0, 1.0 - tpx[3], set_lum(freg_base, rgb2lum(freg_top))); \
    vst1q_f32(bpx, freg_base);                                          \
  } while (0)

/* Four channels RGBA, normalized */
typedef struct imgf32_t imgf32_t;
struct imgf32_t {
  uint32_t width, height;
  float opacity;
  float32_t **rows;
};

typedef struct imgu8_t imgu8_t;
struct imgu8_t {
  uint32_t width, height;
  //float opacity;
  uint8_t **rows;
};

static inline float32x4_t rgb2hsl(float32x4_t rgb) {
  float32_t h, s, l;
  float32_t max = 0.0;
  float32_t min = 1.0;
  float32_t r = vgetq_lane_f32(rgb, 0);
  float32_t g = vgetq_lane_f32(rgb, 1);
  float32_t b = vgetq_lane_f32(rgb, 2);

  max = FFMAX(r, g, b);
  min = FFMIN(r, g, b);
  float32_t risgt = 0.0, gisgt = 0.0, bisgt = 0.0; 
  if (r == max)
    risgt = 1.0;
  else if ( g == max)
    gisgt = 1.0;
  else bisgt = 1.0;

  l = (max + min) / 2.0;
  float32_t xn = (max-min);
  if (xn < 0.001) {
    return (float32x4_t) {0.0, 0.0, l, vgetq_lane_f32(rgb, 3)};
  }
//  fprintf(stderr, "r: %f g: %f b: %f\n", risgt, gisgt, bisgt);

  if (l >= 0.5) {
    s = xn / ((2.0 - max) - min);
  }
  else s = xn / (max + min);
  h = (risgt*( (float32_t)(g < b)*6.0  + ((g-b)/xn))) +
      (gisgt*( 2.0                     + ((b-r)/xn))) +
      (bisgt*( 4.0                     + ((r-g)/xn)));
  h /= 6.0;
//  fprintf(stderr, "r: %f g: %f b: %f\n", r, g, b);
//  fprintf(stderr, "max: %f min: %f\n", max, min);
//  fprintf(stderr, "h: %f s: %f l: %f\n", h, s, l);
  return (float32x4_t) {h, s, l, vgetq_lane_f32(rgb, 3)};
}




static inline float32_t h2rgb(float32_t p, float32_t q, float32_t t) {
  if (t < 0.0) t += 1.0;
  if (t > 1.0) t -= 1.0;
  if (t < 1.0/6.0) return p + (q - p) * 6.0 * t;
  if (t < 1.0/2.0) return q;
  if (t < 2.0/3.0) return p + (q - p) * ((2.0/3.0) - t) * 6.0;
  return p;
}

static inline float32x4_t hsl2rgb(float32x4_t hsl) {
  float32_t h = vgetq_lane_f32(hsl, 0);
  float32_t s = vgetq_lane_f32(hsl, 1);
  float32_t l = vgetq_lane_f32(hsl, 2);

  if (s == 0.0)
    return (float32x4_t){l, l, l, vgetq_lane_f32(hsl, 3)};

  float32_t q = l < 0.5 ? l * (1.0 + s) : l + s - l * s;
  float32_t p = 2.0 * l - q;

  float32_t r = h2rgb(p, q, h + 1.0/3.0);
  float32_t g = h2rgb(p, q, h);
  float32_t b = h2rgb(p, q, h - 1.0/3.0);

  //fprintf(stderr, "r: %f g: %f b: %f\n", r, g, b);
  return (float32x4_t){r, g, b, vgetq_lane_f32(hsl, 3)};
}

    
void free_imgf32(imgf32_t *img) {
  for (uint32_t y = 0; y < img->height; y++)
    free(img->rows[y]);
  free(img->rows);

  free(img);
}

void free_imgu8(imgu8_t *img) {
  for (uint32_t y = 0; y < img->height; y++)
    free(img->rows[y]);
  free(img->rows);

  free(img);
}

/* blend functions. these forward declarations are what functions those DEFINE_BLEND_FUNC()
 * are going to expand to. those macros are discouraged but it keeps this source minimal.
 */

static void blend_BASE(imgf32_t *restrict base, imgf32_t *restrict top);
static void blend_TOP(imgf32_t *restrict base, imgf32_t *restrict top);
static void blend_NORMAL(imgf32_t *restrict base, imgf32_t *restrict top);
static void blend_ADDITION(imgf32_t *restrict base, imgf32_t *restrict top);
static void blend_COLOR(imgf32_t *restrict base, imgf32_t *restrict top);
static void blend_COLOR_DODGE(imgf32_t *restrict base, imgf32_t *restrict top);
static void blend_DIFFERENCE(imgf32_t *restrict base, imgf32_t *restrict top);
static void blend_DARKEN(imgf32_t *restrict base, imgf32_t *restrict top);
static void blend_DIVIDE(imgf32_t *restrict base, imgf32_t *restrict top);
static void blend_GAMMA_LIGHT(imgf32_t *restrict base, imgf32_t *restrict top);
static void blend_GAMMA_DARK(imgf32_t *restrict base, imgf32_t *restrict top);
static void blend_HUE(imgf32_t *restrict base, imgf32_t *restrict top);
static void blend_LIGHTEN(imgf32_t *restrict base, imgf32_t *restrict top);
static void blend_OVERLAY(imgf32_t *restrict base, imgf32_t *restrict top);
static void blend_LUMINOSITY(imgf32_t *restrict base, imgf32_t *restrict top);
static void blend_MULTIPLY(imgf32_t *restrict base, imgf32_t *restrict top);
static void blend_SCREEN(imgf32_t *restrict base, imgf32_t *restrict top);
static void blend_SATURATION(imgf32_t *restrict base, imgf32_t *restrict top);
static void blend_SOFT_LIGHT(imgf32_t *restrict base, imgf32_t *restrict top);
static void blend_HARD_LIGHT(imgf32_t *restrict base, imgf32_t *restrict top);

DEFINE_BLEND_FUNC(BASE);
DEFINE_BLEND_FUNC(TOP);
DEFINE_BLEND_FUNC(NORMAL);
DEFINE_BLEND_FUNC(ADDITION);
DEFINE_BLEND_FUNC(COLOR);
DEFINE_BLEND_FUNC(COLOR_DODGE);
DEFINE_BLEND_FUNC(DIFFERENCE);
DEFINE_BLEND_FUNC(DARKEN);
DEFINE_BLEND_FUNC(DIVIDE);
DEFINE_BLEND_FUNC(GAMMA_LIGHT);
DEFINE_BLEND_FUNC(GAMMA_DARK);
DEFINE_BLEND_FUNC(HUE);
DEFINE_BLEND_FUNC(LIGHTEN);
DEFINE_BLEND_FUNC(OVERLAY);
DEFINE_BLEND_FUNC(LUMINOSITY);
DEFINE_BLEND_FUNC(MULTIPLY);
DEFINE_BLEND_FUNC(SCREEN);
DEFINE_BLEND_FUNC(SATURATION);
DEFINE_BLEND_FUNC(SOFT_LIGHT);
DEFINE_BLEND_FUNC(HARD_LIGHT);

// u8 -> f32
imgf32_t *imgu8_f32(imgu8_t *src) {
  float32_t **dst = malloc(sizeof(*dst) * src->height);
  if (dst == NULL) {
    fprintf(stderr, "no mem\n");
    return NULL;
  }

  for (uint32_t y = 0; y < src->height; y++) {
    dst[y] = malloc(sizeof(**dst) * 4 * src->width);
    assert(dst[y] != NULL);
  }

  for (uint32_t y = 0; y < src->height; y++) {
    float32_t *drow = dst[y];
    uint8_t *srow = src->rows[y];
    for (uint32_t x = 0; x < src->width; x++) {
      float32_t *dpx = drow + x*4;
      uint8_t *spx = srow + x*4;
      float32x4_t freg_src = {
        (float32_t)spx[0], (float32_t)spx[1],
        (float32_t)spx[2], (float32_t)spx[3]
      };
      freg_src = vmulq_n_f32(freg_src, 1.0/255.0);
      vst1q_f32(dpx, freg_src);
    }
  }
  imgf32_t *ret = malloc(sizeof(*ret));
  ret->width = src->width;
  ret->height = src->height;
  ret->rows = dst;
  return ret;
}

// f32 -> u8
imgu8_t *imgf32_u8(imgf32_t *src) {
  uint8_t **dst = malloc(sizeof(*dst) * src->height);
  if (dst == NULL) {
    fprintf(stderr, "no mem\n");
    return NULL;
  }

  for (uint32_t y = 0; y < src->height; y++) {
    dst[y] = malloc(sizeof(**dst) * 4 * src->width);
    assert(dst[y] != NULL);
  }

  for (uint32_t y = 0; y < src->height; y++) {
    uint8_t *drow = dst[y];
    float32_t *srow = src->rows[y];
    for (uint32_t x = 0; x < src->width; x++) {
      uint8_t *(dpx) = (drow + x * 4);
      float32_t *(spx) = (srow + x * 4);
      float32x4_t freg_src = vld1q_f32(spx);
      freg_src = vmulq_n_f32(freg_src, 255.0);
      uint32x4_t freg_dst = vcvtq_u32_f32(freg_src);
      dpx[0] = vgetq_lane_u32(freg_dst, 0) & 0xff;
      dpx[1] = vgetq_lane_u32(freg_dst, 1) & 0xff;
      dpx[2] = vgetq_lane_u32(freg_dst, 2) & 0xff;
      dpx[3] = vgetq_lane_u32(freg_dst, 3) & 0xff;
    }
  }

  imgu8_t *ret = malloc(sizeof(*ret));
  ret->width = src->width;
  ret->height = src->height;
  ret->rows = dst;
  return ret;
}

void write_pngf32(imgf32_t *imgf, FILE *fp) {
  png_structp pstruct =
      png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (pstruct == NULL)
    abort();

  png_infop pinfo = png_create_info_struct(pstruct);
  if (pinfo == NULL)
    abort();

  if (setjmp(png_jmpbuf(pstruct)))
    abort();

  png_init_io(pstruct, fp);

  // Output is 8bit depth, RGBA format.
  png_set_IHDR(pstruct, pinfo, imgf->width, imgf->height, 8, PNG_COLOR_TYPE_RGBA,
               PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);
  png_write_info(pstruct, pinfo);
  png_set_compression_level(pstruct, 3);

  imgu8_t *imgu8 = imgf32_u8(imgf);

  png_write_image(pstruct, imgu8->rows);
  png_write_end(pstruct, NULL);

  png_destroy_write_struct(&pstruct, &pinfo);

  free_imgu8(imgu8);
  fclose(fp);
  return;
}

/* opens png file and stores its value to float32 */
imgf32_t *open_pngf32(char *fstr) {
  float opacity = 1.0;
  char fpname[32] = {0};
  for (unsigned int cndx = 0; cndx < 31 && cndx < strlen(fstr); cndx++) {
    if (fstr[cndx] == ':') {
      strncpy(fpname, fstr, cndx);
      opacity = atof(fstr + cndx + 1);
      break;
    }
  }
  if (!fpname[0]) {
    strncpy(fpname, fstr, 31);
  }

  FILE *fp = fopen(fpname, "rb");
  if (fp == NULL) {
    perror(fpname);
    return NULL; 
  }

  char header[8];
  fread(header, 1, 8, fp);
  if (png_sig_cmp((png_bytep)header, 0, 8)) {
    fprintf(stderr, "%s: not a png\n", fpname);
    return NULL;
  }

  fseek(fp, 0, SEEK_SET);

  png_structp pstruct =
    png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (pstruct == NULL) {
    return NULL;
  }
  
  png_infop pinfo = png_create_info_struct(pstruct);

  if (setjmp(png_jmpbuf(pstruct)))
    abort();

  png_init_io(pstruct, fp);
  png_read_info(pstruct, pinfo);

  uint32_t img_width = png_get_image_width(pstruct, pinfo);
  uint32_t img_height = png_get_image_height(pstruct, pinfo);
  uint32_t img_bit_depth = png_get_bit_depth(pstruct, pinfo);
  uint32_t img_color_type = png_get_color_type(pstruct, pinfo);

  if (img_color_type == PNG_COLOR_TYPE_PALETTE)
    png_set_palette_to_rgb(pstruct);

  if (img_bit_depth == 16)
    png_set_strip_16(pstruct);

  if (img_color_type == PNG_COLOR_TYPE_GRAY && img_bit_depth < 8)
    png_set_expand_gray_1_2_4_to_8(pstruct);

  if (png_get_valid(pstruct, pinfo, PNG_INFO_tRNS))
    png_set_tRNS_to_alpha(pstruct);

  if (img_color_type == PNG_COLOR_TYPE_RGB    ||
      img_color_type == PNG_COLOR_TYPE_GRAY   ||
      img_color_type == PNG_COLOR_TYPE_PALETTE)
    png_set_filler(pstruct, 0xFF, PNG_FILLER_AFTER);

  if (img_color_type == PNG_COLOR_TYPE_GRAY ||
      img_color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
    png_set_gray_to_rgb(pstruct);

  png_read_update_info(pstruct, pinfo);

  uint8_t **img_rows = malloc(sizeof(*img_rows)*img_height);
  assert(img_rows != NULL);

  for (uint32_t y = 0; y < img_height; y++) {
    img_rows[y] = malloc(sizeof(**img_rows)*img_width*4);
    assert(img_rows[y] != NULL);
  }

  png_read_image(pstruct, img_rows);

  imgf32_t *imgf32 = imgu8_f32(&(imgu8_t){
    img_width,
    img_height,
    img_rows
  });
  imgf32->opacity = opacity;


#ifdef FFDEBUG
  for (int x = 0; x < 10; x++) {
    float32_t *fpx = imgf32->rows[0] + x*4;
    //float32x4_t thsl = rgb2hsl((float32x4_t){fpx[0], fpx[1], fpx[2]});
    fprintf(stderr, "t r %f g %f b %f\n", fpx[0], fpx[1], fpx[2]);
//    fprintf(stderr, "t h %f s %f l %f\n",
//     vgetq_lane_f32(thsl, 0),
//     vgetq_lane_f32(thsl, 1),
//     vgetq_lane_f32(thsl, 2));
    fprintf(stderr, "t r %f g %f g %f\n\n",
    vgetq_lane_f32(trgb, 0),
    vgetq_lane_f32(trgb, 1),
    vgetq_lane_f32(trgb, 2));
    
  }
#endif /* FFDEBUG */

  png_destroy_read_struct(&pstruct, &pinfo, NULL);
  fclose(fp);
  return imgf32;  
}

int main(int argc, char *argv[]) {
  (void)rgb2hsl((float32x4_t){2.0});

  int  ret = 1;
  if (argc == 2 && !strcmp(argv[1], "license")) {
    fprintf(stderr, 
       "fflaten Copyright (C) 2024 Minato Yoshie\n"
       "\n"
       "fflatten is free software; you can redistribute it and/or modify it\n"
       "under the terms of the GNU General Public License as published by the\n"
       "Free Software Foundation; either version 2 of the License, or (at your\n"
       "option) any later version.\n"
       "\n"
       "This program is distributed in the hope that it will be useful, but\n"
       "WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY\n"
       "or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License\n"
       "for more details.\n"
       "\n"
       "You should have received a copy of the GNU General Public License along\n"
       "with this program; if not, write to the Free Software Foundation, 51\n"
       "Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.\n\n"
   );
   return 0;
  }
  if (argc & 1) {
    fprintf(stderr, "fflatten "
#ifdef FFLATTEN_INTRINSICS_USED
      "with"
#else
      "without"
#endif
    " intrinscs \n");
    fprintf(stderr, "usage: %s base.png[:opacity] (<operator> top.png[:opacity])*\n", argv[0]);
    fprintf(stderr, "usage: %s license\n", argv[0]);
    fprintf(stderr, "operator:\n");
    PRINT_BLEND_OP(BASE       );
    PRINT_BLEND_OP(TOP        );
    PRINT_BLEND_OP(NORMAL     );
    PRINT_BLEND_OP(ADDITION   );
    PRINT_BLEND_OP(COLOR      );
    PRINT_BLEND_OP(COLOR_DODGE);
    PRINT_BLEND_OP(DARKEN     );
    PRINT_BLEND_OP(DIVIDE     );
    PRINT_BLEND_OP(GAMMA_LIGHT);
    PRINT_BLEND_OP(GAMMA_DARK );
    PRINT_BLEND_OP(HUE        );
    PRINT_BLEND_OP(LIGHTEN    );
    PRINT_BLEND_OP(LUMINOSITY );
    PRINT_BLEND_OP(MULTIPLY   );
    PRINT_BLEND_OP(OVERLAY    );
    PRINT_BLEND_OP(SATURATION );
    PRINT_BLEND_OP(SCREEN     );
    PRINT_BLEND_OP(SOFT_LIGHT );
    PRINT_BLEND_OP(HARD_LIGHT );
    fprintf(stderr, "<required> [optional]\n");
    return 1; 
  }

  imgf32_t *base_img = open_pngf32(argv[1]);
  imgf32_t *top_img = NULL;
  if (base_img == NULL)
      goto clean;

  for (int cur = 2; cur < argc; ) {
    char op = argv[cur++][0];
    top_img = open_pngf32(argv[cur++]);
    if (top_img == NULL)
      goto clean;
  
    if (base_img->width != top_img->width ||
        base_img->height != top_img->height) {
      fprintf(stderr, "base and top must have the same width and height\n");
      goto clean;
    }
  
    switch (op) {
    DEFINE_BLEND_CASE(BASE       ,base_img, top_img);
    DEFINE_BLEND_CASE(TOP        ,base_img, top_img);
    DEFINE_BLEND_CASE(NORMAL     ,base_img, top_img);
    DEFINE_BLEND_CASE(ADDITION   ,base_img, top_img);
    DEFINE_BLEND_CASE(COLOR      ,base_img, top_img);
    DEFINE_BLEND_CASE(COLOR_DODGE,base_img, top_img);
    DEFINE_BLEND_CASE(DIFFERENCE ,base_img, top_img);
    DEFINE_BLEND_CASE(DARKEN     ,base_img, top_img);
    DEFINE_BLEND_CASE(DIVIDE     ,base_img, top_img);
    DEFINE_BLEND_CASE(GAMMA_LIGHT,base_img, top_img);
    DEFINE_BLEND_CASE(GAMMA_DARK ,base_img, top_img);
    DEFINE_BLEND_CASE(HUE        ,base_img, top_img);
    DEFINE_BLEND_CASE(LIGHTEN    ,base_img, top_img);
    DEFINE_BLEND_CASE(LUMINOSITY ,base_img, top_img);
    DEFINE_BLEND_CASE(MULTIPLY   ,base_img, top_img);
    DEFINE_BLEND_CASE(OVERLAY    ,base_img, top_img);
    DEFINE_BLEND_CASE(SATURATION ,base_img, top_img);
    DEFINE_BLEND_CASE(SCREEN     ,base_img, top_img);
    DEFINE_BLEND_CASE(SOFT_LIGHT ,base_img, top_img);
    DEFINE_BLEND_CASE(HARD_LIGHT ,base_img, top_img);
    default:
      fprintf(stderr, "invalid op '%c'\n", op);
      goto clean;
    }

    fprintf(stderr, "'%c' -> %s\n", op, argv[cur - 1]);
    free_imgf32(top_img);
    top_img = NULL;
  }
  write_pngf32(base_img, stdout);

#ifdef FFDEBUG

  for (int x = 0; x < 10; x++) {
    float32_t *dpx = base_img->rows[0] + x*4;
    float32_t *spx = top_img->rows[0] + x*4;
  }

#endif /* FFDEBUG */

clean:
  if (base_img != NULL)
    free_imgf32(base_img);
  if (top_img != NULL)
    free_imgf32(top_img);

  return ret;
}
