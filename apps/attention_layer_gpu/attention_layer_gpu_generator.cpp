#include "Halide.h"

namespace {

using namespace Halide;

const int SCHEDULE = 0;

class AttentionLayerGPU : public Halide::Generator<AttentionLayerGPU> {
public:
  Input <Buffer<float, 3>> input{"input"};
  Input <Buffer<float, 3>> weight_q{"weight_q"};
  Input <Buffer<float, 3>> weight_k{"weight_k"};
  Input <Buffer<float, 3>> weight_v{"weight_v"};
  Output <Buffer<float, 4>> output{"output"};

  // B: Batch size
  // H: Head number
  // N: Token number
  // D: Hidden layer size
  const int B = 8, H = 8, N = 2048, D = 96;

  void generate() {
    /* THE ALGORITHM */
    prod_q(b, h, n, s, d) = input(b, n, d) * weight_q(h, d, s);
    prod_k(b, h, n, s, d) = input(b, n, d) * weight_k(h, d, s);
    prod_v(b, h, n, s, d) = input(b, n, d) * weight_v(h, d, s);

    mat_q(b, h, n, s) += prod_q(b, h, n, s, ddim);
    mat_k(b, h, n, s) += prod_k(b, h, n, s, ddim);
    mat_v(b, h, n, s) += prod_v(b, h, n, s, ddim);

    prod_qkt(b, h, nq, nk, s) = mat_q(b, h, nq, s) * mat_k(b, h, nk, s);

    mat_qkt(b, h, nq, nk) += prod_qkt(b, h, nq, nk, sdim);

    exp_max(b, h, nq) = maximum(mat_qkt(b, h, nq, ndim));
    expo(b, h, nq, nk) = exp(mat_qkt(b, h, nq, nk) - exp_max(b, h, nq));
    normalizer(b, h, nq) += expo(b, h, nq, ndim);
    softmax(b, h, nq, nk) = expo(b, h, nq, nk) / normalizer(b, h, nq);

    prod_sv(b, h, nq, s, n) = softmax(b, h, nq, n) * mat_v(b, h, n, s);
    mat_sv(b, h, n, s) += prod_sv(b, h, n, s, ndim);

    output(b,h, n, s) = mat_sv(b,h, n, s);
  }

  void schedule() {
    /* THE SCHEDULE */
    if (SCHEDULE == 0) {
      prod_q.compute_root();
      mat_q.compute_root();
      prod_k.compute_root();
      mat_k.compute_root();
      prod_v.compute_root();
      mat_v.compute_root();
      prod_qkt.compute_root();
      mat_qkt.compute_root();
      softmax.compute_root();
      mat_sv.compute_root();
      expo.compute_root();
      exp_max.compute_root();
      normalizer.compute_root();
    } else if (SCHEDULE == 1) {
      // fused qkt and softmax at each qkt row
      prod_q.compute_root();
      mat_q.compute_root();
      prod_k.compute_root();
      mat_k.compute_root();
      prod_v.compute_root();
      mat_v.compute_root();
      softmax.compute_root();
      mat_sv.compute_root();
      softmax.reorder(nk, nq, h, b);
      expo.reorder(nk, nq, h, b);
      exp_max.reorder(nq, h, b);
      mat_qkt.reorder(nk, nq, h, b);
      prod_qkt.reorder(s, nk, nq, h, b);
      normalizer.compute_at(softmax, nq);
      expo.compute_at(softmax, nq);
      exp_max.compute_at(softmax, nq);
      mat_qkt.compute_at(softmax, nq);
      prod_qkt.compute_at(mat_qkt, nk);
    } else if (SCHEDULE == 2) {
      // fused qkt , softmax and sv at each qkt row
      prod_q.compute_root();
      mat_q.compute_root();
      prod_k.compute_root();
      mat_k.compute_root();
      prod_v.compute_root();
      mat_v.compute_root();
      softmax.compute_root();
      mat_sv.compute_root();
      softmax.reorder(nk, nq, h, b);
      expo.reorder(nk, nq, h, b);
      exp_max.reorder(nq, h, b);
      mat_qkt.update(0).reorder(nk, nq, h, b);
      prod_qkt.reorder(s, nk, nq, h, b);
      normalizer.compute_at(softmax, nq);
      expo.compute_at(softmax, nq);
      exp_max.compute_at(softmax, nq);
      mat_qkt.compute_at(softmax, nq);
      prod_qkt.compute_at(mat_qkt, nk);
      prod_sv.reorder(n, s, nq, h, b);
      prod_sv.compute_at(mat_sv, s);
      mat_sv.update(0).reorder(s, n, h, b);
      softmax.compute_at(mat_sv, n);
    } else if (SCHEDULE == 3) {
      prod_q.compute_root();
      mat_q.compute_root();
      prod_k.compute_root();
      mat_k.compute_root();
      prod_v.compute_root();
      mat_v.compute_root();
      prod_qkt.compute_root();
      mat_qkt.compute_root();
      softmax.compute_root();
      mat_sv.compute_root();
      expo.compute_root();
      exp_max.compute_root();
      normalizer.compute_root();
      prod_q.parallel(h);
      prod_k.parallel(h);
      prod_v.parallel(h);
      prod_qkt.parallel(h);
    }

  }

private:
  Var b{"b"}, h{"h"}, n{"n"}, d{"d"}, s{"ss"};
  Var nq{"nq"}, nk{"nk"};
  Var c{"c"};
  RDom ddim{0, D*H};
  RDom sdim{0, D};
  RDom ndim{0, N};

  Func prod_q{"prod_q"};
  Func prod_k{"prod_k"};
  Func prod_v{"prod_v"};
  Func mat_q{"mat_q"};
  Func mat_k{"mat_k"};
  Func mat_v{"mat_v"};
  Func prod_qkt{"prod_qkt"};
  Func mat_qkt{"mat_qkt"};
  Func exp_max{"exp_max"}, expo{"expo"}, normalizer{"normalizer"};
  Func softmax{"softmax"};
  Func prod_sv{"prod_sv"};
  Func mat_sv{"mat_sv"};
};

}  // namespace

HALIDE_REGISTER_GENERATOR(AttentionLayerGPU, attention_layer_gpu)
