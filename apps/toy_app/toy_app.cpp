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
        prod_k(n1, s1, d1) = input(n1, d1) * weight_k(d1, s1);

        mat_q(n4, s4) += prod_q(n4, s4, ddim0);
        mat_k(n5, s5) += prod_k(n5, s5, ddim1);

        mat_k_heap(p0 ,q0) = mat_k(p0, q0);
        prod_qkt(nq0, nk0, ss) = mat_q(nq0, ss) * mat_k_heap(nk0, ss);

        mat_qkt(nq2, nk2) += prod_qkt(nq2, nk2, sdim0);

        output(nq3, nk3) = mat_qkt(nq3, nk3);
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
          Var po{"po"}, pi{"pi"};
          Var qo{"qo"}, qi{"qi"};
          Var nqo{"nqo"}, nqi{"nqi"}, nko{"nko"}, nki{"nki"};
          RVar sdimo{"sdimo"}, sdimi{"sdimi"};
          RVar ddimo{"ddimo"}, ddimi{"ddimi"};
          mat_qkt.compute_root();
          mat_qkt.update(0).tile(nq2, nk2, nqo, nko, 1, 1);
          mat_qkt.update(0).gpu_blocks(nq2);
          mat_qkt.update(0).gpu_threads(nk2);
          mat_qkt.update(0).split(sdim0, sdim0, sdimi, 2).reorder(sdimi, nko, nqo, sdim0, nk2, nq2);
          mat_q.compute_at(mat_qkt, nq2);
          mat_q.update(0).split(ddim0, ddimo, ddimi, 2);
          mat_q.update(0).tile(s4, n4, so, no, 1, 1);
          mat_q.update(0).reorder(ddimi, so, no, s4, n4, ddimo);
          mat_q.update(0).gpu_threads(s4, n4);
//          Func mat_k_heap = mat_k.in(prod_qkt);
          mat_k_heap.compute_root();
          mat_k_heap.tile(p0, q0, po, qo, 64, 64);
          mat_k_heap.gpu_blocks(p0, q0);
          mat_k_heap.reorder(po, qo, p0, q0);
          mat_k.compute_at(mat_k_heap, p0);
//          mat_k.store_in(Halide::MemoryType::GPUShared);
          mat_k.update(0).tile(n5, s5, no, so, 16, 16)
              .split(ddim1, ddimo, ddimi, 16)
              .gpu_threads(no, so)
              .reorder(ddimi, no, so, n5, s5, ddimo);
        }

        output.print_loop_nest();
    }

private:
    Var n0{"n0"}, d0{"d0"}, s0{"s0"}, p0{"p0"}, q0{"q0"};
    Var n1{"n1"}, d1{"d1"}, s1{"s1"}, p1{"p1"}, q1{"q1"};
    Var n2{"n2"}, d2{"d2"}, s2{"s2"}, p2{"p2"}, q2{"q2"};
    Var n3{"n3"}, d3{"d3"}, s3{"s3"}, p3{"p3"}, q3{"q3"};
    Var n4{"n4"}, d4{"d4"}, s4{"s4"}, p4{"p4"}, q4{"q4"};
    Var n5{"n5"}, d5{"d5"}, s5{"s5"}, p5{"p5"}, q5{"q5"};
    Var nq0{"nq0"}, nk0{"nk0"};
    Var nq1{"nq1"}, nk1{"nk1"};
    Var nq2{"nq2"}, nk2{"nk2"};
    Var nq3{"nq3"}, nk3{"nk3"};
    Var ss{"ss"};

    RDom ddim0{0, D, "ddim0"};
    RDom sdim0{0, S, "sdim0"};
    RDom ddim1{0, D, "ddim1"};
    RDom sdim1{0, S, "sdim1"};
    RDom ddim2{0, D, "ddim2"};
    RDom sdim2{0, S, "sdim2"};

    Func prod_q{"prod_q"};
    Func prod_k{"prod_k"};
    Func mat_q{"mat_q"};
    Func mat_k{"mat_k"};
    Func prod_qkt{"prod_qkt"};
    Func mat_qkt{"mat_qkt"};
    Func mat_k_heap{"mat_heap"};
};

}  // namespace

HALIDE_REGISTER_GENERATOR(ToyApp, toy_app)
