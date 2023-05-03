#include "Halide.h"

namespace {

using namespace Halide;

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
        const int B = 1, H = 1, N = 512, D = 768, S = 512;

        /* THE ALGORITHM */

        Var b("b"), h("h"), n("n"), d("d"), s("s");

        Func prod_q("prod_q");
        Func prod_k("prod_k");
        Func prod_v("prod_v");

        prod_q(b, h, n, s, d) = input(b, n, d) * weight_q(h, d, s);
        prod_k(b, h, n, s, d) = input(b, n, d) * weight_k(h, d, s);
        prod_v(b, h, n, s, d) = input(b, n, d) * weight_v(h, d, s);

        Func mat_q("mat_q");
        Func mat_k("mat_k");
        Func mat_v("mat_v");

        RDom ddim(0, D);
        mat_q(b, h, n, s) += prod_q(b, h, n, s, ddim);
        mat_k(b, h, n, s) += prod_k(b, h, n, s, ddim);
        mat_v(b, h, n, s) += prod_v(b, h, n, s, ddim);

        Func prod_qkt("prod_qkt");
        Var nq("nq"), nk("nk");
        prod_qkt(b, h, nq, nk, s) = mat_q(b, h, nq, s) * mat_k(b, h, nk, s);

        Func mat_qkt("mat_qkt");
        RDom sdim(0, S);
        mat_qkt(b, h, nq, nk) += prod_qkt(b, h, nq, nk, sdim);

        Func softmax("softmax");
        Func exp_max("exp_max"), expo("expo"), normalizer("normalizer");
        RDom ndim(0, N);
        exp_max(b, h, nq) = maximum(mat_qkt(b, h, nq, ndim));
        expo(b, h, nq, nk) = exp(mat_qkt(b, h, nq, nk) - exp_max(b, h, nq));
        normalizer(b, h, nq) += expo(b, h, nq, ndim);
        softmax(b, h, nq, nk) = expo(b, h, nq, nk) / normalizer(b, h, nq);

        Func mat_sv("mat_sv");
        Func prod_sv("prod_sv");
        prod_sv(b, h, nq, s, n) = softmax(b, h, nq, n) * mat_v(b, h, n, s);
        mat_sv(b, h, n, s) += prod_sv(b, h, n, s, ndim);

        Func concat("concat");
        Var c("c");
        concat(b, n, c) = mat_sv(b, c / S, n, c - c / S);

        Func mat_ao("mat_ao");
        Func prod_ao("prod_ao");
        prod_ao(b, n, d, c) = concat(b, n, c) * weight_o(c, d);
        RDom cdim(0, S * H);
        mat_ao(b, n, d) += prod_ao(b, n, d, cdim);

        output(b, n, d) = mat_ao(b, n, d);

        /* THE SCHEDULE */
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(AttentionLayer, attention_layer)
