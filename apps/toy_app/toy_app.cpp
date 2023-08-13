#include "Halide.h"

namespace {

using namespace Halide;

const int SCHEDULE = 4;

class ToyApp : public Halide::Generator<ToyApp> {
public:
    Input<Buffer<float, 2>> input{"input"};
    Input<Buffer<float, 2>> weight_q{"weight_q"};
    Input<Buffer<float, 2>> weight_k{"weight_k"};
    Output<Buffer<float, 2>> output{"output"};

    // N: Token number
    // D: Token dimension
    // S: Hidden layer size
    const int N = 512, D = 64, S = 256;

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
        }

        mat_qkt.print_loop_nest();
    }

private:
    Var n{"n"}, d{"d"}, s{"s"};
    Var nq{"nq"}, nk{"nk"};
    RDom ddim{0, D};
    RDom sdim{0, S};
    RDom ndim{0, N};

    Func prod_q{"prod_q"};
    Func prod_k{"prod_k"};
    Func mat_q{"mat_q"};
    Func mat_k{"mat_k"};
    Func prod_qkt{"prod_qkt"};
    Func mat_qkt{"mat_qkt"};
};

}  // namespace

HALIDE_REGISTER_GENERATOR(ToyApp, toy_app)
