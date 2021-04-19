// A collection of utility functions shared by the halide generators.

#ifndef HANNK_COMMON_HALIDE_H
#define HANNK_COMMON_HALIDE_H

#include "Halide.h"

namespace hannk {

using Halide::Internal::rounding_shift_right;
using Halide::Internal::saturating_add;
using Halide::Internal::saturating_sub;
using Halide::Internal::widening_add;
using Halide::Internal::widening_mul;
using Halide::Internal::widening_sub;

// Get the number of vector registers available on the target.
int get_register_count(const Halide::Target &target);

// A tensor has the same requirements as a buffer in Halide by default, except
// the min of the innermost dimension must also be 0.
void interpret_as_tensor(Halide::OutputImageParam p);

// Require dimension dim have the same min and extent.
void require_same_min_extent(int dim, Halide::OutputImageParam first, Halide::OutputImageParam second);

Halide::Expr is_interleaved(Halide::OutputImageParam p, int channels);

// Round x down or up to the nearest multiple of n.
Halide::Expr align_down(const Halide::Expr &x, const Halide::Expr &n);
Halide::Expr align_up(const Halide::Expr &x, const Halide::Expr &n);
Halide::Expr align(const Halide::Expr &x, const Halide::Expr &n);

// Compute saturating_narrow(rounding_shift_right(widening_mul(a, b), N))
// where N is the number of bits of the narrowed result minus one.
Halide::Expr multiply_2x_high(const Halide::Expr &a, const Halide::Expr &b);

// Approximate log2(x)*2^log2_precision.
// log2_precision must be less than 16.
Halide::Expr approx_log2(const Halide::Expr &x, int log2_precision);
Halide::Expr approx_log2(const Halide::Type &type, const Halide::Expr &x, int log2_precision);

// Approximate 2^(x/2^log2_precision_x)/2^log2_precision_result.
// log2_precision_x must be less than 16.
Halide::Expr approx_exp2(const Halide::Expr &x, const Halide::Expr &log2_precision_x, int log2_precision_result);
Halide::Expr approx_exp2(const Halide::Type &type, const Halide::Expr &x, const Halide::Expr &log2_precision_x, int log2_precision_result);

// Approximate 2^log2_precision/x
Halide::Expr approx_reciprocal(const Halide::Expr &x, int log2_precision);

// Approximate 2^log2_precision/sqrt(x)
Halide::Expr approx_reciprocal_sqrt(const Halide::Expr &x, int log2_precision);

}  // namespace hannk

#endif  // HANNK_COMMON_HALIDE_H