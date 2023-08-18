#include "Halide.h"

namespace {

using namespace Halide;

const int SCHEDULE = 17;

class ToyApp : public Halide::Generator<ToyApp> {
public:
    Input<Buffer<float, 2>> input{"input"};
    Input<Buffer<float, 2>> weight_q{"weight_q"};
    Input<Buffer<float, 2>> weight_k{"weight_k"};
    Output<Buffer<float, 2>> output{"output"};

    // N: Token number
    // D: Token dimension
    // S: Hidden layer size
    const int N = 512, D = 512, S = 256;

    void generate() {
        /* THE ALGORITHM */
        prod_q(n, s, d) = input(n, d) * weight_q(d, s);
        prod_k(n, s, d) = input(n, d) * weight_k(d, s);

        mat_q(n, s) += prod_q(n, s, ddim);
        mat_k(n, s) += prod_k(n, s, ddim);

        prod_qkt(nq, nk, s) = mat_q(nq, s) * mat_k(nk, s);

        mat_qkt(nq, nk) += prod_qkt(nq, nk, sdim);

        output(nq, nk) = mat_qkt(nq, nk);
    }

    void schedule() {
        /* THE SCHEDULE */
        if (SCHEDULE == 0) {
            prod_q.compute_root();
            mat_q.compute_root();
            prod_k.compute_root();
            mat_k.compute_root();
            prod_qkt.compute_root();
            mat_qkt.compute_root();
        } else if (SCHEDULE == 1) {
            prod_q.reorder(d, s, n);
            mat_q.reorder(s, n);
            prod_k.reorder(d, s, n);
            mat_k.reorder(s, n);
            prod_qkt.reorder(s, nk, nq);
            mat_qkt.reorder(nk, nq);
            prod_q.compute_root();
            mat_q.compute_root();
            prod_k.compute_root();
            mat_k.compute_root();
            prod_qkt.compute_root();
            mat_qkt.compute_root();
        } else if (SCHEDULE == 3) {
          Var so, si, no, ni;
          Var nqo, nqi, nko, nki;
          prod_q.reorder(d, s, n);
          mat_q.reorder(s, n);
          prod_k.reorder(d, s, n);
          mat_k.reorder(s, n);
          prod_qkt.reorder(s, nk, nq);
          mat_qkt.reorder(nk, nq);
          prod_q.compute_root();
          prod_q.gpu_tile(s, n, so, no, si, ni, 8, 8);
          mat_q.compute_root();
          mat_q.update(0).gpu_tile(s, n, so, no, si, ni, 8, 8);
          prod_k.compute_root();
          prod_k.gpu_tile(s, n, so, no, si, ni, 8, 8);
          mat_k.compute_root();
          mat_k.update(0).gpu_tile(s, n, so, no, si, ni, 8, 8);
          prod_qkt.compute_root();
          prod_qkt.gpu_tile(nq, nk, nqo, nko, nqi, nki, 8, 8);
          mat_qkt.compute_root();
          mat_qkt.update(0).gpu_tile(nq, nk, nqo, nko, nqi, nki, 8, 8);
        } else if (SCHEDULE == 4) {
          Var so, si, no, ni;
          Var nqo, nqi, nko, nki;
          prod_q.reorder(d, s, n);
          mat_q.reorder(s, n);
          prod_k.reorder(d, s, n);
          mat_k.reorder(s, n);
          prod_qkt.reorder(s, nk, nq);
          mat_qkt.reorder(nk, nq);
          prod_q.compute_root();
          prod_q.gpu_tile(s, n, so, no, si, ni, 16, 16);
          mat_q.compute_root();
          mat_q.update(0).gpu_tile(s, n, so, no, si, ni, 16, 16);
          prod_k.compute_root();
          prod_k.gpu_tile(s, n, so, no, si, ni, 16, 16);
          mat_k.compute_root();
          mat_k.update(0).gpu_tile(s, n, so, no, si, ni, 16, 16);
          prod_qkt.compute_root();
          prod_qkt.gpu_tile(nq, nk, nqo, nko, nqi, nki, 16, 16);
          mat_qkt.compute_root();
          mat_qkt.update(0).gpu_tile(nq, nk, nqo, nko, nqi, nki, 16, 16);
        } else if (SCHEDULE == 5) {
          Var so, si, no, ni;
          Var nqo, nqi, nko, nki;
          prod_q.reorder(d, s, n);
          mat_q.reorder(s, n);
          prod_k.reorder(d, s, n);
          mat_k.reorder(s, n);
          prod_qkt.reorder(s, nk, nq);
          mat_qkt.reorder(nk, nq);
          prod_q.compute_root();
          prod_q.gpu_tile(s, n, so, no, si, ni, 32, 32);
          mat_q.compute_root();
          mat_q.update(0).gpu_tile(s, n, so, no, si, ni, 32, 32);
          prod_k.compute_root();
          prod_k.gpu_tile(s, n, so, no, si, ni, 32, 32);
          mat_k.compute_root();
          mat_k.update(0).gpu_tile(s, n, so, no, si, ni, 32, 32);
          prod_qkt.compute_root();
          prod_qkt.gpu_tile(nq, nk, nqo, nko, nqi, nki, 32, 32);
          mat_qkt.compute_root();
          mat_qkt.update(0).gpu_tile(nq, nk, nqo, nko, nqi, nki, 32, 32);
        } else if (SCHEDULE == 6) {
          Var so, si, no, ni;
          Var nqo, nqi, nko, nki;
          prod_q.reorder(d, s, n);
          mat_q.reorder(s, n);
          prod_k.reorder(d, s, n);
          mat_k.reorder(s, n);
          prod_qkt.reorder(s, nk, nq);
          mat_qkt.reorder(nk, nq);
          prod_q.compute_root();
          prod_q.gpu_tile(s, n, so, no, si, ni, 8, 8);
          mat_q.compute_root();
          mat_q.update(0).gpu_tile(s, n, so, no, si, ni, 8, 8);
          mat_q.store_in(Halide::MemoryType::GPUShared);
          prod_k.compute_root();
          prod_k.gpu_tile(s, n, so, no, si, ni, 8, 8);
          mat_k.compute_root();
          mat_k.update(0).gpu_tile(s, n, so, no, si, ni, 8, 8);
          mat_k.store_in(Halide::MemoryType::GPUShared);
          prod_qkt.compute_root();
          prod_qkt.gpu_tile(nq, nk, nqo, nko, nqi, nki, 8, 8);
          mat_qkt.compute_root();
          mat_qkt.update(0).gpu_tile(nq, nk, nqo, nko, nqi, nki, 8, 8);
        } else if (SCHEDULE == 7) {
          Var so, si, no, ni, d_o, di;
          Var nqo, nqi, nko, nki;
          output.compute_root();
          mat_qkt.compute_root();
          mat_qkt.gpu_tile(nq, nk, nqo, nko, nqi, nki, 32, 32);
          mat_qkt.reorder(nqi, nki, nqo, nko);
          mat_qkt.update(0).gpu_tile(nq, nk, nqo, nko, nqi, nki, 32, 32);
          mat_qkt.update(0).reorder(nqi, nki, nqo, nko);
          prod_qkt.compute_at(mat_qkt, nqi);
          prod_qkt.reorder(nq, nk, s);
          mat_q.compute_at(prod_qkt, nq);
          mat_q.update(0).reorder(n, s);
          prod_q.compute_at(mat_q, n);
          prod_q.reorder(d, n, s);
          mat_k.compute_at(prod_qkt, nk);
          mat_k.update(0).reorder(n, s);
          prod_k.compute_at(mat_k, n);
          prod_k.reorder(d, n, s);
        } else if (SCHEDULE == 8) {
          Var so, si, no, ni, d_o, di;
          Var nqo, nqi, nko, nki;
          output.compute_root();
          mat_qkt.compute_root();
          mat_qkt.gpu_tile(nq, nk, nqo, nko, nqi, nki, 32, 32);
          mat_qkt.reorder(nqi, nki, nqo, nko);
          mat_qkt.update(0).gpu_tile(nq, nk, nqo, nko, nqi, nki, 32, 32);
          mat_qkt.update(0).reorder(nqi, nki, nqo, nko);
          prod_qkt.compute_at(mat_qkt, nqi);
          prod_qkt.reorder(nq, nk, s);
          mat_q.compute_root();
          mat_q.update(0).gpu_tile(n, s, no, so, ni, si, 16, 16);
          mat_q.update(0).reorder(ni, si, no, so);
          prod_q.compute_at(mat_q, ni);
          prod_q.reorder(d, n, s);
          mat_k.compute_at(prod_qkt, nk);
          mat_k.update(0).reorder(n, s);
          prod_k.compute_at(mat_k, n);
          prod_k.reorder(d, n, s);
        } else if (SCHEDULE == 9) {
          Var so, si, no, ni, d_o, di;
          Var nqo, nqi, nko, nki;
          output.compute_root();
          mat_qkt.compute_root();
          mat_qkt.gpu_tile(nq, nk, nqo, nko, nqi, nki, 32, 32);
          mat_qkt.reorder(nqi, nki, nqo, nko);
          mat_qkt.update(0).gpu_tile(nq, nk, nqo, nko, nqi, nki, 32, 32);
          mat_qkt.update(0).reorder(nqi, nki, nqo, nko);
          prod_qkt.compute_at(mat_qkt, nqi);
          prod_qkt.reorder(nq, nk, s);
          mat_q.compute_root();
          mat_q.update(0).gpu_tile(n, s, no, so, ni, si, 16, 16);
          mat_q.update(0).reorder(ni, si, no, so);
          prod_q.compute_at(mat_q, ni);
          prod_q.reorder(d, n, s);
          mat_k.compute_root();
          mat_k.update(0).gpu_tile(n, s, no, so, ni, si, 16, 16);
          mat_k.update(0).reorder(ni, si, no, so);
          prod_k.compute_at(mat_k, ni);
          prod_k.reorder(d, n, s);
        } else if (SCHEDULE == 10) {
          Var so, si, no, ni, d_o, di;
          Var nqo, nqi, nko, nki;
          output.compute_root();
          mat_qkt.compute_root();
          mat_qkt.gpu_tile(nq, nk, nqo, nko, nqi, nki, 32, 32);
          mat_qkt.reorder(nqi, nki, nqo, nko);
          mat_qkt.update(0).tile(nq, nk, nqo, nko, 32, 32);
          mat_qkt.update(0).tile(nqo, nko, nqi, nki, 4, 4);
          mat_qkt.update(0).gpu_blocks(nq, nk);
          mat_qkt.update(0).gpu_threads(nqo, nko);
          mat_qkt.unroll(nqi);
          mat_qkt.unroll(nki);
          mat_qkt.update(0).reorder(nqi, nki, nqo, nko, nq, nk);
          prod_qkt.compute_at(mat_qkt, nqi);
          prod_qkt.reorder(nq, nk, s);
          mat_q.compute_root();
          mat_q.update(0).tile(n, s, no, so, 16, 16);
          mat_q.update(0).tile(no, so, ni, si, 4, 4);
          mat_q.update(0).gpu_blocks(n, s);
          mat_q.update(0).gpu_threads(no, so);
          mat_q.update(0).unroll(ni);
          mat_q.update(0).unroll(si);
          mat_q.update(0).reorder(ni, si, no, so, n, s);
          prod_q.compute_at(mat_q, ni);
          prod_q.reorder(d, n, s);
          mat_k.compute_root();
          mat_k.update(0).tile(n, s, no, so, 16, 16);
          mat_k.update(0).tile(no, so, ni, si, 4, 4);
          mat_k.update(0).gpu_blocks(n, s);
          mat_k.update(0).gpu_threads(no, so);
          mat_k.update(0).unroll(ni);
          mat_k.update(0).unroll(si);
          mat_k.update(0).reorder(ni, si, no, so, n, s);
          prod_k.compute_at(mat_k, ni);
          prod_k.reorder(d, n, s);
        } else if (SCHEDULE == 11) {
          Var so, si, no, ni, d_o, di;
          Var nqo, nqi, nko, nki;
          Var nqii, nkii;
          output.compute_root();
          mat_qkt.compute_root();
          mat_qkt.gpu_tile(nq, nk, nqo, nko, nqi, nki, 32, 32);
          mat_qkt.reorder(nqi, nki, nqo, nko);
          mat_qkt.update(0).tile(nq, nk, nqo, nko, 64, 64);
          mat_qkt.update(0).tile(nqo, nko, nqi, nki, 16, 16);
          mat_qkt.update(0).gpu_blocks(nqo, nko);
          mat_qkt.update(0).gpu_threads(nqi, nki);
          mat_qkt.update(0).reorder(nqi, nki, nqo, nko, nq, nk);
          prod_qkt.compute_at(mat_qkt, nqi);
          prod_qkt.reorder(s, nq, nk);
          mat_q.compute_root();
          mat_q.update(0).tile(n, s, no, so, 16, 16);
          mat_q.update(0).tile(no, so, ni, si, 4, 4);
          mat_q.update(0).gpu_blocks(n, s);
          mat_q.update(0).gpu_threads(no, so);
          mat_q.update(0).unroll(ni);
          mat_q.update(0).unroll(si);
          mat_q.update(0).reorder(ni, si, no, so, n, s);
          prod_q.compute_at(mat_q, ni);
          prod_q.reorder(d, n, s);
          mat_k.compute_root();
          mat_k.update(0).tile(n, s, no, so, 16, 16);
          mat_k.update(0).tile(no, so, ni, si, 4, 4);
          mat_k.update(0).gpu_blocks(n, s);
          mat_k.update(0).gpu_threads(no, so);
          mat_k.update(0).unroll(ni);
          mat_k.update(0).unroll(si);
          mat_k.update(0).reorder(ni, si, no, so, n, s);
          prod_k.compute_at(mat_k, ni);
          prod_k.reorder(d, n, s);
        } else if (SCHEDULE == 12) {
          Var so, si, no, ni, d_o, di;
          Var nqo, nqi, nko, nki;
          RVar sdimo, sdimi;
          output.compute_root();
          mat_qkt.compute_root();
          mat_qkt.gpu_tile(nq, nk, nqo, nko, nqi, nki, 32, 32);
          mat_qkt.reorder(nqi, nki, nqo, nko);
          mat_qkt.update(0).tile(nq, nk, nqo, nko, 64, 64);
          mat_qkt.update(0).tile(nqo, nko, nqi, nki, 16, 16);
          mat_qkt.update(0).split(sdim, sdimo, sdimi, 16);
          mat_qkt.update(0).gpu_blocks(nqo, nko);
          mat_qkt.update(0).gpu_threads(nqi, nki);
          mat_qkt.update(0).reorder(sdimi, nqi, nki, sdimo, nqo, nko, nq, nk);
          prod_qkt.compute_at(mat_qkt, nqi);
          prod_qkt.reorder(s, nq, nk);
          mat_q.compute_root();
          mat_q.update(0).tile(n, s, no, so, 16, 16);
          mat_q.update(0).tile(no, so, ni, si, 4, 4);
          mat_q.update(0).gpu_blocks(n, s);
          mat_q.update(0).gpu_threads(no, so);
          mat_q.update(0).reorder(ni, si, no, so, n, s);
          prod_q.compute_at(mat_q, ni);
          prod_q.reorder(d, n, s);
          mat_k.compute_root();
          mat_k.update(0).tile(n, s, no, so, 16, 16);
          mat_k.update(0).tile(no, so, ni, si, 4, 4);
          mat_k.update(0).gpu_blocks(n, s);
          mat_k.update(0).gpu_threads(no, so);
          mat_k.update(0).reorder(ni, si, no, so, n, s);
          prod_k.compute_at(mat_k, ni);
          prod_k.reorder(d, n, s);
        } else if (SCHEDULE == 13) {
          Var so, si, no, ni, d_o, di;
          Var nqo, nqi, nko, nki;
          RVar sdimo, sdimi;
          output.compute_root();
          mat_qkt.compute_root();
          mat_qkt.gpu_tile(nq, nk, nqo, nko, nqi, nki, 32, 32);
          mat_qkt.reorder(nqi, nki, nqo, nko);
          mat_qkt.update(0).tile(nq, nk, nqo, nko, 256, 256);
          mat_qkt.update(0).tile(nqo, nko, nqi, nki, 16, 16);
          mat_qkt.update(0).split(sdim, sdimo, sdimi, 16);
          mat_qkt.update(0).gpu_blocks(nqo, nko);
          mat_qkt.update(0).gpu_threads(nqi, nki);
          mat_qkt.update(0).reorder(sdimi, nqi, nki, sdimo, nqo, nko, nq, nk);
          prod_qkt.compute_at(mat_qkt, sdimo);
          prod_qkt.gpu_threads(nq, nk);
          prod_qkt.reorder(s, nq, nk);
          prod_qkt.store_in(Halide::MemoryType::GPUShared);
          mat_q.compute_root();
          mat_q.update(0).tile(n, s, no, so, 16, 16);
          mat_q.update(0).tile(no, so, ni, si, 4, 4);
          mat_q.update(0).gpu_blocks(n, s);
          mat_q.update(0).gpu_threads(no, so);
          mat_q.update(0).reorder(ni, si, no, so, n, s);
          prod_q.compute_at(mat_q, ni);
          prod_q.reorder(d, n, s);
          mat_k.compute_root();
          mat_k.update(0).tile(n, s, no, so, 16, 16);
          mat_k.update(0).tile(no, so, ni, si, 4, 4);
          mat_k.update(0).gpu_blocks(n, s);
          mat_k.update(0).gpu_threads(no, so);
          mat_k.update(0).reorder(ni, si, no, so, n, s);
          prod_k.compute_at(mat_k, ni);
          prod_k.reorder(d, n, s);
        } else if (SCHEDULE == 14) {
          Var so{"so"}, si{"si"}, no{"no"}, ni{"ni"}, d_o{"do"}, di{"di"};
          Var nqo{"nqo"}, nqi{"nqi"}, nko{"nko"}, nki{"nki"};
          RVar sdimo{"sdimo"}, sdimi{"sdimi"};
          RVar ddimo{"ddimo"}, ddimi{"ddimi"};
          output.compute_root();
          output.tile(nq, nk, nqo, nko, 8, 8);
          output.reorder(nko, nqo, nk, nq);
          output.gpu_blocks(nk, nq);
          mat_qkt.compute_at(output, nk);
          mat_qkt.update(0).split(sdim, sdim, sdimo, 8).split(sdimo, sdimo, sdimi, 4).reorder(sdimi, sdimo, nk, nq, sdim);
          prod_qkt.compute_at(mat_qkt, sdim);
          prod_qkt.gpu_threads(nq, nk);
          prod_qkt.reorder(nq, nk, s);
          mat_q.compute_at(output, nk);
          mat_q.update(0).split(ddim, ddimo, ddimi, 8);
          mat_q.update(0).reorder(ddimi, s, n, ddimo);
          prod_q.compute_at(mat_q, ddimo);
          prod_q.gpu_threads(s, n);
          prod_q.reorder(d, s, n);
          mat_k.compute_root();
          mat_k.update(0).tile(n, s, no, so, 16, 16)
              .gpu_blocks(n, s)
              .gpu_threads(no, so)
              .split(ddim, ddimo, ddimi, 16)
              .reorder(no, so, ddimi, ddimo, n, s);
          prod_k.compute_at(mat_k, ddimi);
          prod_k.gpu_threads(n, s);
          prod_k.reorder(n, s, d);
        } else if (SCHEDULE == 15) {
          Var so{"so"}, si{"si"}, no{"no"}, ni{"ni"}, d_o{"do"}, di{"di"};
          Var nqo{"nqo"}, nqi{"nqi"}, nko{"nko"}, nki{"nki"};
          RVar sdimo{"sdimo"}, sdimi{"sdimi"};
          RVar ddimo{"ddimo"}, ddimi{"ddimi"};
          output.compute_root();
          mat_qkt.compute_root();
          mat_qkt.gpu_tile(nq, nk, nqo, nko, nqi, nki, 32, 32);
          mat_qkt.reorder(nqi, nki, nqo, nko);
          mat_qkt.update(0).split(nq, nqo, nqi, 16);
          mat_qkt.update(0).split(nk, nko, nki, 16);
          mat_qkt.update(0).split(sdim, sdimo, sdimi, 16);
          mat_qkt.update(0).gpu_blocks(nqo);
          mat_qkt.update(0).gpu_threads(nki);
          mat_qkt.update(0).gpu_threads(nqi);
          mat_qkt.update(0).reorder(sdimi, nqi, nki, nko, sdimo, nqo);
          prod_qkt.compute_at(mat_qkt, nko);
          prod_qkt.gpu_threads(nq, nk);
          prod_qkt.reorder(s, nq, nk);
          prod_qkt.store_in(Halide::MemoryType::GPUShared);
          mat_k.in(prod_qkt).store_in(Halide::MemoryType::GPUShared);
          mat_q.compute_at(mat_qkt, nko);
          mat_q.store_in(Halide::MemoryType::GPUShared);
          mat_q.gpu_threads(n, s);
          mat_q.update(0).split(ddim, ddimo, ddimi, 4);
          mat_q.update(0).gpu_threads(s, n);
          mat_q.update(0).reorder(ddimi, s, n, ddimo);
          prod_q.compute_at(mat_q, ddimo);
          prod_q.store_in(Halide::MemoryType::GPUShared);
          prod_q.gpu_threads(s, n);
          prod_q.reorder(s, n, d);
          input.in(prod_q).store_in(Halide::MemoryType::GPUShared);
          weight_q.in(prod_q).store_in(Halide::MemoryType::GPUShared);
          mat_k.compute_root();
          mat_k.gpu_tile(n, s, no, so, 16, 16)
               .reorder(no, so, n, s);
          mat_k.update(0).tile(n, s, no, so, 16, 16)
                         .gpu_blocks(n, s)
                         .gpu_threads(no, so)
                         .split(ddim, ddimo, ddimi, 8)
                         .reorder(ddimi, no, so, ddimo, n, s);
          prod_k.compute_at(mat_k, ddimo);
          prod_k.store_in(Halide::MemoryType::GPUShared);
          prod_k.gpu_threads(d, n, s);
          prod_k.reorder(d, n, s);
          input.in(prod_k).store_in(Halide::MemoryType::GPUShared);
          weight_k.in(prod_k).store_in(Halide::MemoryType::GPUShared);
        } else if (SCHEDULE == 16) {
          Var so{"so"}, si{"si"}, no{"no"}, ni{"ni"}, d_o{"do"}, di{"di"};
          Var nqo{"nqo"}, nqi{"nqi"}, nko{"nko"}, nki{"nki"};
          RVar sdimo{"sdimo"}, sdimi{"sdimi"};
          RVar ddimo{"ddimo"}, ddimi{"ddimi"};
          mat_qkt.compute_root();
          mat_qkt.update(0).tile(nq, nk, nqo, nko, 16, 16);
          mat_qkt.update(0).gpu_blocks(nk, nq);
          mat_qkt.update(0).split(sdim, sdim, sdimi, 16).reorder(sdimi, nko, nqo, sdim, nk, nq);
          prod_qkt.compute_at(mat_qkt, sdim);
          prod_qkt.gpu_threads(nq, nk);
          prod_qkt.reorder(nq, nk, s);
          mat_q.compute_at(mat_qkt, nk);
          mat_q.update(0).split(ddim, ddimo, ddimi, 16);
          mat_q.update(0).tile(s, n, so, no, 16, 16);
          mat_q.update(0).reorder(ddimi, so, no, s, n, ddimo);
          mat_q.update(0).gpu_threads(so, no);
          prod_q.compute_at(mat_q, s);
          prod_q.store_in(Halide::MemoryType::Heap);
          prod_q.split(s, so, si, 16);
          prod_q.gpu_threads(si, n);
          prod_q.reorder(si, n, so, d);
          mat_k.compute_root();
          mat_k.update(0).tile(n, s, no, so, 16, 16)
              .gpu_blocks(n, s)
              .gpu_threads(no, so)
              .split(ddim, ddimo, ddimi, 16)
              .reorder(no, so, ddimi, ddimo, n, s);
          prod_k.compute_at(mat_k, ddimi);
          prod_k.gpu_threads(n, s);
          prod_k.reorder(n, s, d);
        } else if (SCHEDULE == 17) {
          Var so{"so"}, si{"si"}, no{"no"}, ni{"ni"}, d_o{"do"}, di{"di"};
          Var nqo{"nqo"}, nqi{"nqi"}, nko{"nko"}, nki{"nki"};
          RVar sdimo{"sdimo"}, sdimi{"sdimi"};
          RVar ddimo{"ddimo"}, ddimi{"ddimi"};
          mat_qkt.compute_root();
          mat_qkt.update(0).tile(nq, nk, nqo, nko, 4, 16);
          mat_qkt.update(0).gpu_blocks(nq);
          mat_qkt.update(0).gpu_threads(nk);
          mat_qkt.update(0).split(sdim, sdim, sdimi, 16).reorder(sdimi, nko, nqo, sdim, nk, nq);
          prod_qkt.compute_at(mat_qkt, sdim);
          prod_qkt.gpu_threads(nk);
          prod_qkt.reorder(nq, nk, s);
          mat_q.compute_at(mat_qkt, nq);
          mat_q.update(0).split(ddim, ddimo, ddimi, 16);
          mat_q.update(0).tile(s, n, so, no, 2, 2);
          mat_q.update(0).reorder(ddimi, so, no, s, n, ddimo);
          mat_q.update(0).gpu_threads(s, n);
          prod_q.compute_at(mat_q, s);
          mat_k.compute_root();
          mat_k.update(0).tile(n, s, no, so, 16, 16)
              .gpu_blocks(n, s)
              .gpu_threads(no, so)
              .split(ddim, ddimo, ddimi, 16)
              .reorder(no, so, ddimi, ddimo, n, s);
          prod_k.compute_at(mat_k, ddimi);
          prod_k.gpu_threads(n, s);
          prod_k.reorder(n, s, d);
        }

        output.print_loop_nest();
    }

private:
    Var n{"n"}, d{"d"}, s{"s"};
    Var nq{"nq"}, nk{"nk"};
    RDom ddim{0, D, "ddim"};
    RDom sdim{0, S, "sdim"};
    RDom ndim{0, N, "ndim"};

    Func prod_q{"prod_q"};
    Func prod_k{"prod_k"};
    Func mat_q{"mat_q"};
    Func mat_k{"mat_k"};
    Func prod_qkt{"prod_qkt"};
    Func mat_qkt{"mat_qkt"};
};

}  // namespace

HALIDE_REGISTER_GENERATOR(ToyApp, toy_app)
