#ifndef COMPAT_H
#define COMPAT_H

#ifdef _MSC_VER
#include <windows.h>
#include <math.h>
#include <stdlib.h>

// MSVC doesn't have roundf, but it has round.
// We can define roundf to be round.
#define roundf(x) round(x)

// MSVC doesn't define M_PI, so we define it here.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// MSVC doesn't have fmaxf, but it has fmax.
// We can define fmaxf to be fmax.
#define fmaxf(x, y) fmax(x, y)

// Undefine ERROR if it's defined, to avoid conflicts with wingdi.h
#ifdef ERROR
#undef ERROR
#endif

// MSVC doesn't support C99 complex numbers, so we define our own
typedef struct { float real, imag; } complex_float;
#define I _Complex_I
static inline float crealf(complex_float c) { return c.real; }
static inline float cimagf(complex_float c) { return c.imag; }
static inline complex_float _Cbuildf(float r, float i) { complex_float c = {r, i}; return c; }

#else
#include <complex.h>
#include <math.h>
typedef float _Complex complex_float;
#define _Cbuildf(r, i) ((complex_float)((r) + (i) * _Complex_I))
#endif

#endif // COMPAT_H
