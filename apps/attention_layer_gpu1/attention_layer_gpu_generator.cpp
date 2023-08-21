#include "Halide.h"

namespace {

using namespace Halide;

const int SCHEDULE = 0;

class AttentionLayerGPU1 : public Halide::Generator<AttentionLayerGPU1> {
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

    output.dim(0).set_extent(B);
    output.dim(1).set_extent(H);
    output.dim(2).set_extent(N);
    output.dim(3).set_extent(D);

    input.dim(0).set_extent(B);
    input.dim(1).set_extent(N);
    input.dim(2).set_extent(H * D);

    weight_k.dim(0).set_extent(H);
    weight_k.dim(1).set_extent(H * D);
    weight_k.dim(2).set_extent(D);
    weight_q.dim(0).set_extent(H);
    weight_q.dim(1).set_extent(H * D);
    weight_q.dim(2).set_extent(D);
    weight_v.dim(0).set_extent(H);
    weight_v.dim(1).set_extent(H * D);
    weight_v.dim(2).set_extent(D);

  }

  void schedule() {
    /* THE SCHEDULE */
    if (SCHEDULE == 0) {
      mat_sv.in(output).compute_root();
      mat_sv.in(output).tile(b, h, bo, ho, bi, hi, 1, 1);
      mat_sv.in(output).tile(ss, n, sso, no, ssi, ni, 32, 32);
      mat_sv.in(output).tile(ssi, ni, ssi, ni, sst, nt, 32 / 8, 32 / 4);
      mat_sv.in(output).reorder(nt, sst, ssi, ni, bi, hi, no, ho, bo, sso);
      mat_sv.in(output).gpu_blocks(no, ho, bo);
      mat_sv.in(output).gpu_threads(ssi, ni);
      mat_sv.in(output).vectorize(nt, 8);
      mat_sv.in(output).unroll(sst, 8);
      mat_sv.compute_at(mat_sv.in(output), no);
      mat_sv.update(0).tile(b, h, bo, ho, bi, hi, 1, 1);
      mat_sv.update(0).split(ndim, ndimo, ndimi, 1);
      mat_sv.update(0).tile(ss, n, sso, no, ssi, ni, 32, 32);
      mat_sv.update(0).tile(ssi, ni, ssi, ni, sst, nt, 32 / 8, 32 / 4);
      mat_sv.update(0).reorder(nt, sst, ndimi, ssi, ni, bi, hi, ndimo, no, ho, bo, sso);
      mat_sv.update(0).gpu_threads(ssi, ni);
      mat_sv.update(0).vectorize(nt, 8);
      mat_sv.update(0).unroll(sst);
      mat_qkt.in(prod_sv).in(prod_sv).compute_at(mat_sv, ndimo);
      mat_qkt.in(prod_sv).in(prod_sv).gpu_threads(nq);
      mat_v.in(prod_sv).in(prod_sv).compute_at(mat_sv, ndimo);
      mat_v.in(prod_sv).in(prod_sv).gpu_threads(ss);

      mat_qkt.in(prod_sv).compute_root();
      mat_qkt.in(prod_sv).tile(b, h, bo, ho, bi, hi, 1, 1);
      mat_qkt.in(prod_sv).tile(nk, nq, nko, nqo, nki, nqi, 32, 32);
      mat_qkt.in(prod_sv).tile(nki, nqi, nki, nqi, nkt, nqt, 32 / 8, 32 / 4);
      mat_qkt.in(prod_sv).reorder(nqt, nkt, nqi, nki, bi, hi, nqo, ho, bo, nko);
      mat_qkt.in(prod_sv).gpu_blocks(nko, ho, bo);
      mat_qkt.in(prod_sv).gpu_threads(nki, nqi);
      mat_qkt.in(prod_sv).vectorize(nqt, 8);
      mat_qkt.in(prod_sv).unroll(nkt);
      mat_qkt.compute_at(mat_qkt.in(prod_sv), nqo);
      mat_qkt.update(0).tile(b, h, bo, ho, bi, hi, 1, 1);
      mat_qkt.update(0).split(sdim, sdimo, sdimi, 3);
      mat_qkt.update(0).tile(nk, nq, nko, nqo, nki, nqi, 32, 32);
      mat_qkt.update(0).tile(nki, nqi, nki, nqi, nkt, nqt, 32 / 8, 32 / 4);
      mat_qkt.update(0).reorder(nqt, nkt, sdimi, nqi, nki, bi, hi, sdimo, nqo, ho, bo, nko);
      mat_qkt.update(0).gpu_threads(nki, nqi);
      mat_qkt.update(0).vectorize(nqt, 8);
      mat_qkt.update(0).unroll(nkt);
      mat_qkt.update(0).unroll(sdimi);
      mat_k.in(prod_qkt).in(prod_qkt).compute_at(mat_qkt, ho);
      mat_k.in(prod_qkt).in(prod_qkt).fuse(ss, n, n);
      mat_k.in(prod_qkt).in(prod_qkt).split(n, no, ni, 32);
      mat_k.in(prod_qkt).in(prod_qkt).gpu_threads(ni);
      mat_q.in(prod_qkt).in(prod_qkt).compute_at(mat_qkt, sdimo);
      mat_q.in(prod_qkt).in(prod_qkt).fuse(ss, n, n);
      mat_q.in(prod_qkt).in(prod_qkt).split(n, no, ni, 32);
      mat_q.in(prod_qkt).in(prod_qkt).gpu_threads(ni);

      mat_v.in(prod_sv).compute_root();
      mat_v.in(prod_sv).tile(b, h, bo, ho, bi, hi, 1, 1);
      mat_v.in(prod_sv).tile(n, ss, no, sso, ni, ssi, 32, 16);
      mat_v.in(prod_sv).tile(ni, ssi, ni, ssi, nt, sst, 32 / 16, 16 / 4);
      mat_v.in(prod_sv).reorder(nt, sst, ni, ssi, bi, hi, ho, sso, bo, no);
      mat_v.in(prod_sv).gpu_blocks(bo, no);
      mat_v.in(prod_sv).gpu_threads(ni, ssi);
      mat_v.in(prod_sv).vectorize(nt);
      mat_v.in(prod_sv).unroll(sst);
      mat_v.compute_at(mat_v.in(prod_sv), sso);
      mat_v.update(0).tile(b, h, bo, ho, bi, hi, 1, 1);
      mat_v.update(0).split(ddim, ddimo, ddimi, 1);
      mat_v.update(0).tile(n, ss, no, sso, ni, ssi, 32, 16);
      mat_v.update(0).tile(ni, ssi, ni, ssi, nt, sst, 32 / 16, 16 / 4);
      mat_v.update(0).reorder(nt, sst, ddimi, ni, ssi, bi, hi, ho, ddimo, sso, bo, no);
      mat_v.update(0).gpu_threads(ni, ssi);
      mat_v.update(0).vectorize(nt);
      mat_v.update(0).unroll(sst);
      input.in(prod_v).compute_at(mat_v, ho);
      input.in(prod_v).gpu_threads(_1);
      weight_v.in(prod_v).compute_at(mat_v, ho);
      weight_v.in(prod_v).gpu_threads(_2);

      mat_q.in(prod_qkt).compute_root();
      mat_q.in(prod_qkt).tile(b, h, bo, ho, bi, hi, 1, 1);
      mat_q.in(prod_qkt).tile(n, ss, no, sso, ni, ssi, 32, 16);
      mat_q.in(prod_qkt).tile(ni, ssi, ni, ssi, nt, sst, 32 / 16, 16 / 4);
      mat_q.in(prod_qkt).reorder(nt, sst, ni, ssi, bi, hi, ho, sso, bo, no);
      mat_q.in(prod_qkt).gpu_blocks(bo, no);
      mat_q.in(prod_qkt).gpu_threads(ni, ssi);
      mat_q.in(prod_qkt).vectorize(nt);
      mat_q.in(prod_qkt).unroll(sst);
      mat_q.compute_at(mat_q.in(prod_qkt), sso);
      mat_q.update(0).tile(b, h, bo, ho, bi, hi, 1, 1);
      mat_q.update(0).split(ddim, ddimo, ddimi, 1);
      mat_q.update(0).tile(n, ss, no, sso, ni, ssi, 32, 16);
      mat_q.update(0).tile(ni, ssi, ni, ssi, nt, sst, 32 / 16, 16 / 4);
      mat_q.update(0).reorder(nt, sst, ddimi, ni, ssi, bi, hi, ho, ddimo, sso, bo, no);
      mat_q.update(0).gpu_threads(ni, ssi);
      mat_q.update(0).vectorize(nt);
      mat_q.update(0).unroll(sst);
      input.in(prod_q).compute_at(mat_q, ho);
      input.in(prod_q).gpu_threads(_1);
      weight_q.in(prod_q).compute_at(mat_q, ho);
      weight_q.in(prod_q).gpu_threads(_2);

      mat_k.in(prod_qkt).compute_root();
      mat_k.in(prod_qkt).tile(b, h, bo, ho, bi, hi, 1, 1);
      mat_k.in(prod_qkt).tile(n, ss, no, sso, ni, ssi, 32, 16);
      mat_k.in(prod_qkt).tile(ni, ssi, ni, ssi, nt, sst, 16 / 8, 16 / 4);
      mat_k.in(prod_qkt).reorder(nt, sst, ni, ssi, bi, hi, ho, sso, bo, no);
      mat_k.in(prod_qkt).gpu_blocks(bo, no);
      mat_k.in(prod_qkt).gpu_threads(ni, ssi);
      mat_k.in(prod_qkt).vectorize(nt);
      mat_k.in(prod_qkt).unroll(sst);
      mat_k.compute_at(mat_k.in(prod_qkt), sso);
      mat_k.update(0).tile(b, h, bo, ho, bi, hi, 1, 1);
      mat_k.update(0).split(ddim, ddimo, ddimi, 1);
      mat_k.update(0).tile(n, ss, no, sso, ni, ssi, 32, 16);
      mat_k.update(0).tile(ni, ssi, ni, ssi, nt, sst, 16 / 8, 16 / 4);
      mat_k.update(0).reorder(nt, sst, ddimi, ni, ssi, bi, hi, ho, ddimo, sso, bo, no);
      mat_k.update(0).gpu_threads(ni, ssi);
      mat_k.update(0).vectorize(nt);
      mat_k.update(0).unroll(sst);
      input.in(prod_k).compute_at(mat_k, ho);
      input.in(prod_k).gpu_threads(_1);
      weight_k.in(prod_k).compute_at(mat_k, ho);
      weight_k.in(prod_k).gpu_threads(_2);

    } else if (SCHEDULE == 5) {
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

HALIDE_REGISTER_GENERATOR(AttentionLayerGPU1, attention_layer_gpu1)
