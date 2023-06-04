#include "Halide.h"

namespace {

using namespace Halide;

const int SCHEDULE = 0;

inline void apply_schedule_attention_layer_auto_schedule(
    ::Halide::Pipeline pipeline
) {
    using ::Halide::Func;
    using ::Halide::MemoryType;
    using ::Halide::RVar;
    using ::Halide::TailStrategy;
    using ::Halide::Var;
    Func output = pipeline.get_func(23);
    Func mat_ao = pipeline.get_func(22);
    Func prod_ao = pipeline.get_func(21);
    Func concat = pipeline.get_func(19);
    Func mat_sv = pipeline.get_func(18);
    Func prod_sv = pipeline.get_func(17);
    Func softmax = pipeline.get_func(16);
    Func normalizer = pipeline.get_func(15);
    Func expo = pipeline.get_func(14);
    Func exp_max = pipeline.get_func(13);
    Func maximum = pipeline.get_func(12);
    Func mat_qkt = pipeline.get_func(11);
    Func prod_qkt = pipeline.get_func(10);
    Func mat_q = pipeline.get_func(9);
    Func prod_q = pipeline.get_func(8);
    Func mat_k = pipeline.get_func(6);
    Func prod_k = pipeline.get_func(5);
    Func mat_v = pipeline.get_func(3);
    Func prod_v = pipeline.get_func(2);
    Var b(output.get_schedule().dims()[0].var);
    Var bi("bi");
    Var c(concat.get_schedule().dims()[2].var);
    Var ci("ci");
    Var d(output.get_schedule().dims()[2].var);
    Var di("di");
    Var h(mat_sv.get_schedule().dims()[1].var);
    Var n(output.get_schedule().dims()[1].var);
    Var ni("ni");
    Var nk(softmax.get_schedule().dims()[3].var);
    Var nki("nki");
    Var nq(softmax.get_schedule().dims()[2].var);
    Var nqi("nqi");
    Var s(mat_sv.get_schedule().dims()[3].var);
    Var si("si");
    Var sii("sii");
    RVar r18_x(mat_q.update(0).get_schedule().dims()[0].var);
    RVar r23_x(mat_qkt.update(0).get_schedule().dims()[0].var);
    RVar r28_x(mat_sv.update(0).get_schedule().dims()[0].var);
    RVar r33_x(mat_ao.update(0).get_schedule().dims()[0].var);
    output
        .split(b, b, bi, 16, TailStrategy::GuardWithIf)
        .vectorize(bi)
        .compute_root()
        .reorder({bi, b, d, n})
        .parallel(n);
    mat_ao
        .split(d, d, di, 16, TailStrategy::RoundUp)
        .vectorize(di)
        .compute_at(output, n)
        .reorder({di, d, b, n})
        .reorder_storage(d, b, n);
    mat_ao.update(0)
        .split(d, d, di, 16, TailStrategy::GuardWithIf)
        .vectorize(di)
        .reorder({di, r33_x, d, b, n});
    concat
        .split(c, c, ci, 16, TailStrategy::RoundUp)
        .vectorize(ci)
        .compute_at(output, n)
        .reorder({ci, c, b, n})
        .reorder_storage(c, b, n);
    mat_sv
        .split(s, s, si, 16, TailStrategy::RoundUp)
        .vectorize(si)
        .compute_at(output, n)
        .reorder({si, s, b, h, n})
        .reorder_storage(s, b, h, n);
    mat_sv.update(0)
        .split(s, s, si, 16, TailStrategy::RoundUp)
        .vectorize(si)
        .reorder({si, r28_x, s, b, h, n});
    softmax
        .split(nk, nk, nki, 16, TailStrategy::RoundUp)
        .vectorize(nki)
        .compute_at(output, n)
        .reorder({nki, nk, b, h, nq})
        .reorder_storage(nk, b, h, nq);
    normalizer
        .split(nq, nq, nqi, 16, TailStrategy::RoundUp)
        .vectorize(nqi)
        .compute_root()
        .reorder({nqi, nq, b, h})
        .parallel(nq)
        .reorder_storage(nq, b, h);
    normalizer.update(0)
        .split(nq, nq, nqi, 16, TailStrategy::RoundUp)
        .vectorize(nqi)
        .reorder({nqi, r28_x, nq, b, h})
        .parallel(nq);
    expo
        .split(nq, nq, nqi, 16, TailStrategy::RoundUp)
        .split(nk, nk, nki, 16, TailStrategy::RoundUp)
        .vectorize(nki)
        .compute_root()
        .reorder({nki, nk, nqi, b, h, nq})
        .parallel(nq)
        .reorder_storage(nk, b, h, nq);
    exp_max
        .split(nq, nq, nqi, 16, TailStrategy::RoundUp)
        .vectorize(nqi)
        .compute_at(expo, b)
        .reorder({nqi, nq, b, h})
        .reorder_storage(nq, b, h);
    maximum
        .split(nq, nq, nqi, 16, TailStrategy::RoundUp)
        .vectorize(nqi)
        .compute_at(expo, b)
        .reorder({nqi, nq, b, h})
        .reorder_storage(nq, b, h);
    maximum.update(0)
        .split(nq, nq, nqi, 16, TailStrategy::RoundUp)
        .vectorize(nqi)
        .reorder({nqi, r28_x, nq, b, h});
    mat_qkt
        .split(nk, nk, nki, 16, TailStrategy::RoundUp)
        .vectorize(nki)
        .compute_at(expo, b)
        .reorder({nki, nk, b, h, nq})
        .reorder_storage(nk, b, h, nq);
    mat_qkt.update(0)
        .split(nk, nk, nki, 16, TailStrategy::RoundUp)
        .vectorize(nki)
        .reorder({nki, r23_x, nk, b, h, nq});
    mat_q
        .split(s, s, si, 128, TailStrategy::RoundUp)
        .split(n, n, ni, 32, TailStrategy::RoundUp)
        .split(si, si, sii, 16, TailStrategy::RoundUp)
        .vectorize(sii)
        .compute_root()
        .reorder({sii, si, ni, s, b, h, n})
        .fuse(s, n, s)
        .parallel(s)
        .reorder_storage(s, b, h, n);
    mat_q.update(0)
        .split(s, s, si, 128, TailStrategy::GuardWithIf)
        .split(n, n, ni, 32, TailStrategy::GuardWithIf)
        .split(si, si, sii, 16, TailStrategy::GuardWithIf)
        .vectorize(sii)
        .reorder({sii, r18_x, si, ni, s, b, h, n})
        .fuse(s, n, s)
        .parallel(s);
    mat_k
        .split(s, s, si, 4, TailStrategy::RoundUp)
        .split(n, n, ni, 16, TailStrategy::RoundUp)
        .vectorize(ni)
        .compute_root()
        .reorder({ni, n, si, b, h, s})
        .parallel(s)
        .reorder_storage(n, b, h, s);
    mat_k.update(0)
        .split(s, s, si, 4, TailStrategy::GuardWithIf)
        .split(n, n, ni, 16, TailStrategy::GuardWithIf)
        .vectorize(ni)
        .reorder({ni, r18_x, n, si, b, h, s})
        .parallel(s);
    mat_v
        .split(s, s, si, 128, TailStrategy::RoundUp)
        .split(n, n, ni, 32, TailStrategy::RoundUp)
        .split(si, si, sii, 16, TailStrategy::RoundUp)
        .vectorize(sii)
        .compute_root()
        .reorder({sii, si, ni, s, b, h, n})
        .fuse(s, n, s)
        .parallel(s)
        .reorder_storage(s, b, h, n);
    mat_v.update(0)
        .split(s, s, si, 128, TailStrategy::GuardWithIf)
        .split(n, n, ni, 32, TailStrategy::GuardWithIf)
        .split(si, si, sii, 16, TailStrategy::GuardWithIf)
        .vectorize(sii)
        .reorder({sii, r18_x, si, ni, s, b, h, n})
        .fuse(s, n, s)
        .parallel(s);

}

class AttentionLayer : public Halide::Generator<AttentionLayer> {
public:
    Input<Buffer<float, 3>> input{"input"};
    Input<Buffer<float, 3>> weight_q{"weight_q"};
    Input<Buffer<float, 3>> weight_k{"weight_k"};
    Input<Buffer<float, 3>> weight_v{"weight_v"};
    Input<Buffer<float, 2>> weight_o{"weight_o"};
    Output<Buffer<float, 3>> output{"output"};

    // B: Batch size
    // H: Head number
    // N: Token number
    // D: Token dimension
    // S: Hidden layer size
    const int B = 1, H = 4, N = 256, D = 512, S = 512;

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

        concat(b, n, c) = mat_sv(b, c / S, n, c % S);

        prod_ao(b, n, d, c) = concat(b, n, c) * weight_o(c, d);
        mat_ao(b, n, d) += prod_ao(b, n, d, cdim);

        output(b, n, d) = mat_ao(b, n, d);
    }

    void schedule() {
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
        } else {
            apply_schedule_attention_layer_auto_schedule(get_pipeline());
        }

    }

private:
    Var b{"b"}, h{"h"}, n{"n"}, d{"d"}, s{"s"};
    Var nq{"nq"}, nk{"nk"};
    Var c{"c"};
    RDom ddim{0, D};
    RDom sdim{0, S};
    RDom ndim{0, N};
    RDom cdim{0, S * H};

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
    Func concat{"concat"};
    Func prod_ao{"prod_ao"};
    Func mat_ao{"mat_ao"};
};

}  // namespace

HALIDE_REGISTER_GENERATOR(AttentionLayer, attention_layer)
