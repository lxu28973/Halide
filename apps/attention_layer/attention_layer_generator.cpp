#include "Halide.h"

namespace {

using namespace Halide;

const int SCHEDULE = 0;

class AttentionLayer : public Halide::Generator<AttentionLayer> {
public:
    Input<Buffer<float, 3>> input{"input"};
    Input<Buffer<float, 3>> weight_q{"weight_q"};
    Input<Buffer<float, 3>> weight_k{"weight_k"};
    Input<Buffer<float, 3>> weight_v{"weight_v"};
    Input<Buffer<float, 2>> weight_o{"weight_o"};
    Output<Buffer<float, 3>> output{"output"};

    void generate() {
        // B: Batch size
        // H: Head number
        // N: Token number
        // D: Token dimension
        // S: Hidden layer size
        const Expr B = input.dim(0).extent(), H = weight_v.dim(0).extent(),
                   N = input.dim(1).extent(), D = input.dim(2).extent(),
                   S = weight_v.dim(2).extent();

        /* THE ALGORITHM */

        Var b("b"), h("h"), n("n"), d("d"), s("s");
        Var nq("nq"), nk("nk");
        Var c("c");
        RDom ddim(0, D);
        RDom sdim(0, S);
        RDom ndim(0, N);
        RDom cdim(0, S * H);

        Func prod_q("prod_q");
        Func prod_k("prod_k");
        Func prod_v("prod_v");
        Func mat_q("mat_q");
        Func mat_k("mat_k");
        Func mat_v("mat_v");
        Func prod_qkt("prod_qkt");
        Func mat_qkt("mat_qkt");
        Func exp_max("exp_max"), expo("expo"), normalizer("normalizer");
        Func softmax("softmax");
        Func prod_sv("prod_sv");
        Func mat_sv("mat_sv");
        Func concat("concat");
        Func prod_ao("prod_ao");
        Func mat_ao("mat_ao");

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

        concat(b, n, c) = mat_sv(b, c / S, n, c % S);

        prod_ao(b, n, d, c) = concat(b, n, c) * weight_o(c, d);
        mat_ao(b, n, d) += prod_ao(b, n, d, cdim);

        output(b, n, d) = mat_ao(b, n, d);

        /* THE SCHEDULE */
        if (using_autoscheduler()) {

            input.dim(0).set_estimate(0, 1);
            input.dim(1).set_estimate(0, 256);
            input.dim(2).set_estimate(0, 256);

            weight_q.dim(0).set_estimate(0, 1);
            weight_q.dim(1).set_estimate(0, 256);
            weight_q.dim(2).set_estimate(0, 256);

            weight_k.dim(0).set_estimate(0, 1);
            weight_k.dim(1).set_estimate(0, 256);
            weight_k.dim(2).set_estimate(0, 256);

            weight_v.dim(0).set_estimate(0, 1);
            weight_v.dim(1).set_estimate(0, 256);
            weight_v.dim(2).set_estimate(0, 256);

            weight_o.dim(0).set_estimate(0, 256);
            weight_o.dim(1).set_estimate(0, 256);

            output.dim(0).set_estimate(0, 1);
            output.dim(1).set_estimate(0, 256);
            output.dim(2).set_estimate(0, 256);

        } else if (SCHEDULE == 0) {
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
            mat_ao.compute_root();
            prod_ao.compute_root();
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
            mat_ao.compute_root();
            prod_ao.compute_root();
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
            mat_ao.compute_root();
            prod_ao.compute_root();
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
            mat_ao.compute_root();
            prod_ao.compute_root();
            expo.compute_root();
            exp_max.compute_root();
            normalizer.compute_root();
            prod_q.parallel(h);
            prod_k.parallel(h);
            prod_v.parallel(h);
            prod_qkt.parallel(h);
        }

    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(AttentionLayer, attention_layer)
