#include "Halide.h"

namespace {

using namespace Halide;

const int SCHEDULE = 4;

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
    prod_q(b, h, n, ss, d) = input(b, n, d) * weight_q(h, d, ss);
    prod_k(b, h, n, ss, d) = input(b, n, d) * weight_k(h, d, ss);
    prod_v(b, h, n, ss, d) = input(b, n, d) * weight_v(h, d, ss);

    mat_q(b, h, n, ss) += prod_q(b, h, n, ss, ddim);
    mat_k(b, h, n, ss) += prod_k(b, h, n, ss, ddim);
    mat_v(b, h, n, ss) += prod_v(b, h, n, ss, ddim);

    prod_qkt(b, h, nq, nk, ss) = mat_q(b, h, nq, ss) * mat_k(b, h, nk, ss);

    mat_qkt(b, h, nq, nk) += prod_qkt(b, h, nq, nk, sdim);

    prod_sv(b, h, nq, ss, n) = mat_qkt(b, h, nq, n) * mat_v(b, h, n, ss);
    mat_sv(b, h, n, ss) += prod_sv(b, h, n, ss, ndim);

    output(b, h, n, ss) = mat_sv(b, h, n, ss);
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
      mat_sv.compute_root();
    } else if (SCHEDULE == 1) {
      // fused qkt and softmax at each qkt row
      prod_q.compute_root();
      mat_q.compute_root();
      prod_k.compute_root();
      mat_k.compute_root();
      prod_v.compute_root();
      mat_v.compute_root();
      mat_sv.compute_root();
      mat_qkt.reorder(nk, nq, h, b);
      prod_qkt.reorder(ss, nk, nq, h, b);
      mat_qkt.compute_at(mat_sv, nq);
      prod_qkt.compute_at(mat_qkt, nk);
    } else if (SCHEDULE == 2) {
      // fused qkt , softmax and sv at each qkt row
      prod_q.compute_root();
      mat_q.compute_root();
      prod_k.compute_root();
      mat_k.compute_root();
      prod_v.compute_root();
      mat_v.compute_root();
      mat_sv.compute_root();
      mat_qkt.update(0).reorder(nk, nq, h, b);
      prod_qkt.reorder(ss, nk, nq, h, b);
      mat_qkt.compute_at(mat_sv, nq);
      prod_qkt.compute_at(mat_qkt, nk);
      prod_sv.reorder(n, ss, nq, h, b);
      prod_sv.compute_at(mat_sv, ss);
      mat_sv.update(0).reorder(ss, n, h, b);
    } else if (SCHEDULE == 3) {
      mat_q.compute_root();
      mat_k.compute_root();
      mat_qkt.compute_root();
      mat_v.compute_root();
      mat_sv.compute_root();
    } else if (SCHEDULE == 4) {
      mat_sv.compute_root();
      mat_sv.update(0).tile(b, h, bo, ho, bi, hi, 1, 1);
      mat_sv.update(0).split(ndim, ndimo, ndimi, 1);
      mat_sv.update(0).tile(ss, n, sso, no, ssi, ni, 96, 64);
      mat_sv.update(0).tile(ssi, ni, ssi, ni, sst, nt, 96 / 16, 64 / 16);
      mat_sv.update(0).reorder(bi, hi, ndimi, sst, nt, ssi, ni, ndimo, no, ho, bo, sso);
      mat_sv.update(0).gpu_blocks(ho, bo, sso);
      mat_sv.update(0).gpu_threads(ssi, ni);

      mat_qkt.compute_root();
      mat_qkt.update(0).tile(b, h, bo, ho, bi, hi, 1, 1);
      mat_qkt.update(0).split(sdim, sdimo, sdimi, 3);
      mat_qkt.update(0).tile(nk, nq, nko, nqo, nki, nqi, 64, 64);
      mat_qkt.update(0).tile(nki, nqi, nki, nqi, nkt, nqt, 64 / 16, 64 / 16);
      mat_qkt.update(0).reorder(bi, hi, sdimi, nqt, nkt, nqi, nki, sdimo, nqo, ho, bo, nko);
      mat_qkt.update(0).gpu_blocks(ho, bo, nko);
      mat_qkt.update(0).gpu_threads(nki, nqi);
      mat_k.in(prod_qkt).in(prod_qkt).compute_at(mat_qkt, sdimo);

      mat_v.in(prod_sv).compute_root();
      mat_v.in(prod_sv).tile(b, h, bo, ho, bi, hi, 1, 1);
      mat_v.in(prod_sv).tile(n, ss, no, sso, ni, ssi, 64, 16);
      mat_v.in(prod_sv).tile(ni, ssi, ni, ssi, nt, sst, 64 / 16, 16 / 16);
      mat_v.in(prod_sv).reorder(bi, hi, nt, sst, ni, ssi, ho, sso, bo, no);
      mat_v.in(prod_sv).gpu_blocks(sso, bo, no);
      mat_v.in(prod_sv).gpu_threads(ni, ssi);
      mat_v.compute_at(mat_v.in(prod_sv), sso);
      mat_v.update(0).tile(b, h, bo, ho, bi, hi, 1, 1);
      mat_v.update(0).split(ddim, ddimo, ddimi, 1);
      mat_v.update(0).tile(n, ss, no, sso, ni, ssi, 64, 16);
      mat_v.update(0).tile(ni, ssi, ni, ssi, nt, sst, 64 / 16, 16 / 16);
      mat_v.update(0).reorder(bi, hi, ddimi, nt, sst, ni, ssi, ho, ddimo, sso, bo, no);
      mat_v.update(0).gpu_threads(ni, ssi);

      mat_q.in(prod_qkt).compute_root();
      mat_q.in(prod_qkt).tile(b, h, bo, ho, bi, hi, 1, 1);
      mat_q.in(prod_qkt).tile(n, ss, no, sso, ni, ssi, 64, 16);
      mat_q.in(prod_qkt).tile(ni, ssi, ni, ssi, nt, sst, 64 / 16, 16 / 16);
      mat_q.in(prod_qkt).reorder(bi, hi, nt, sst, ni, ssi, ho, sso, bo, no);
      mat_q.in(prod_qkt).gpu_blocks(sso, bo, no);
      mat_q.in(prod_qkt).gpu_threads(ni, ssi);
      mat_q.compute_at(mat_q.in(prod_qkt), sso);
      mat_q.update(0).tile(b, h, bo, ho, bi, hi, 1, 1);
      mat_q.update(0).split(ddim, ddimo, ddimi, 1);
      mat_q.update(0).tile(n, ss, no, sso, ni, ssi, 64, 16);
      mat_q.update(0).tile(ni, ssi, ni, ssi, nt, sst, 64 / 16, 16 / 16);
      mat_q.update(0).reorder(bi, hi, ddimi, nt, sst, ni, ssi, ho, ddimo, sso, bo, no);
      mat_q.update(0).gpu_threads(ni, ssi);

      mat_k.in(prod_qkt).compute_root();
      mat_k.in(prod_qkt).tile(b, h, bo, ho, bi, hi, 1, 1);
      mat_k.in(prod_qkt).tile(n, ss, no, sso, ni, ssi, 64, 16);
      mat_k.in(prod_qkt).tile(ni, ssi, ni, ssi, nt, sst, 64 / 16, 16 / 16);
      mat_k.in(prod_qkt).reorder(bi, hi, nt, sst, ni, ssi, ho, sso, bo, no);
      mat_k.in(prod_qkt).gpu_blocks(sso, bo, no);
      mat_k.in(prod_qkt).gpu_threads(ni, ssi);
      mat_k.compute_at(mat_k.in(prod_qkt), sso);
      mat_k.update(0).tile(b, h, bo, ho, bi, hi, 1, 1);
      mat_k.update(0).split(ddim, ddimo, ddimi, 1);
      mat_k.update(0).tile(n, ss, no, sso, ni, ssi, 64, 16);
      mat_k.update(0).tile(ni, ssi, ni, ssi, nt, sst, 64 / 16, 16 / 16);
      mat_k.update(0).reorder(bi, hi, ddimi, nt, sst, ni, ssi, ho, ddimo, sso, bo, no);
      mat_k.update(0).gpu_threads(ni, ssi);

//      mat_q.compute_root();
//      mat_q.tile(b, h, bo, ho, bi, hi, 1, 1);
//      mat_q.split(ddim, ddimo, ddimi, 1);
//      mat_q.tile(n, ss, no, sso, ni, ssi, 64, 16);
//      mat_q.reorder(ho, ddimo, sso, bo, no);
//      mat_k.compute_root();
//      mat_k.tile(b, h, bo, ho, bi, hi, 1, 1);
//      mat_k.split(ddim, ddimo, ddimi, 1);
//      mat_k.tile(n, ss, no, sso, ni, ssi, 64, 16);
//      mat_k.reorder(ho, ddimo, sso, bo, no);
    }

    output.print_loop_nest();

  }

private:
  Var b{"b"}, h{"h"}, n{"n"}, d{"d"}, ss{"ss"};
  Var bo{"bo"}, ho{"ho"}, no{"no"}, d_o{"d_o"}, sso{"sso"};
  Var bi{"bi"}, hi{"hi"}, ni{"ni"}, d_i{"d_i"}, ssi{"ssi"};
  Var bt{"bt"}, ht{"ht"}, nt{"nt"}, d_t{"d_t"}, sst{"sst"};
  Var nq{"nq"}, nk{"nk"};
  Var nqo{"nqo"}, nko{"nko"};
  Var nqi{"nqi"}, nki{"nki"};
  Var nqt{"nqt"}, nkt{"nkt"};
  RDom ddim{0, D * H, "ddim"};
  RDom sdim{0, D, "sdim"};
  RDom ndim{0, N, "ndim"};
  RDom ddimo{0, D * H, "ddimo"};
  RDom sdimo{0, D, "sdimo"};
  RDom ndimo{0, N, "ndimo"};
  RDom ddimi{0, D * H, "ddimi"};
  RDom sdimi{0, D, "sdimi"};
  RDom ndimi{0, N, "ndimi"};

  Func prod_q{"prod_q"};
  Func prod_k{"prod_k"};
  Func prod_v{"prod_v"};
  Func mat_q{"mat_q"};
  Func mat_k{"mat_k"};
  Func mat_v{"mat_v"};
  Func prod_qkt{"prod_qkt"};
  Func mat_qkt{"mat_qkt"};
  Func prod_sv{"prod_sv"};
  Func mat_sv{"mat_sv"};
};

}  // namespace

HALIDE_REGISTER_GENERATOR(AttentionLayerGPU, attention_layer_gpu)
