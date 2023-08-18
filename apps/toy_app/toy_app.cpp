#include "Halide.h"

namespace {

using namespace Halide;

const int SCHEDULE = 18;

class ToyApp : public Halide::Generator<ToyApp> {
public:
    Input<Buffer<float, 2>> input{"input"};
    Input<Buffer<float, 2>> weight_q{"weight_q"};
    Input<Buffer<float, 2>> weight_k{"weight_k"};
    Output<Buffer<float, 2>> output{"output"};

    // N: Token number
    // D: Token dimension
    // S: Hidden layer size
    const int N = 512, D = 512, S = 512;

    void generate() {
        /* THE ALGORITHM */
        prod_q(n0, s0, d0) = input(n0, d0) * weight_q(d0, s0);
        prod_k(n0, s0, d0) = input(n0, d0) * weight_k(d0, s0);

        mat_q(n0, s0) += prod_q(n0, s0, ddim0);
        mat_k(n0, s0) += prod_k(n0, s0, ddim0);

        prod_qkt(nq0, nk0, s0) = mat_q(nq0, s0) * mat_k(nk0, s0);

        mat_qkt(nq0, nk0) += prod_qkt(nq0, nk0, sdim0);

        output(nq0, nk0) = mat_qkt(nq0, nk0);
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
        } else if (SCHEDULE == 18) {
          Var so{"so"}, si{"si"}, no{"no"}, ni{"ni"}, d_o{"do"}, di{"di"};
          Var nqo{"nqo"}, nqi{"nqi"}, nko{"nko"}, nki{"nki"};
          RVar sdimo{"sdimo"}, sdimi{"sdimi"};
          RVar ddimo{"ddimo"}, ddimi{"ddimi"};
          mat_qkt.compute_root();
          mat_qkt.update(0).tile(nq0, nk0, nqo, nko, 1, 1);
          mat_qkt.update(0).gpu_blocks(nq0);
          mat_qkt.update(0).gpu_threads(nk0);
          mat_qkt.update(0).split(sdim0, sdim0, sdimi, 2).reorder(sdimi, nko, nqo, sdim0, nk0, nq0);
          mat_q.compute_at(mat_qkt, nq0);
          mat_q.update(0).split(ddim0, ddimo, ddimi, 2);
          mat_q.update(0).tile(s0, n0, so, no, 1, 1);
          mat_q.update(0).reorder(ddimi, so, no, s0, n0, ddimo);
          mat_q.update(0).gpu_threads(s0, n0);
          Func mat_k_heap = mat_k.in(prod_qkt);
          mat_k_heap.compute_root();
          mat_k_heap.tile(n0, s0, no, so, 64, 64);
          mat_k_heap.gpu_blocks(n0, s0);
          mat_k_heap.reorder(no, so, s0, n0);
          mat_k.compute_at(mat_k_heap, s0);
          mat_k.update(0).tile(n0, s0, no, so, 16, 16)
              .split(ddim0, ddimo, ddimi, 16)
              .gpu_threads(no, so)
              .reorder(ddimi, no, so, ddimo, n0, s0);
        }

        output.print_loop_nest();
    }

private:
    Var n0{"n0"}, d0{"d0"}, s0{"s0"}, p0{"p0"}, q0{"q0"};
    Var nq0{"nq0"}, nk0{"nk0"};
    RDom ddim0{0, D, "ddim0"};
    RDom sdim0{0, S, "sdim0"};

    Func prod_q{"prod_q"};
    Func prod_k{"prod_k"};
    Func mat_q{"mat_q"};
    Func mat_k{"mat_k"};
    Func prod_qkt{"prod_qkt"};
    Func mat_qkt{"mat_qkt"};
};

}  // namespace

HALIDE_REGISTER_GENERATOR(ToyApp, toy_app)
