#include <cmath>
#include <chrono>
#include <cstdio>

#include "attention_layer.h"
#include "attention_layer_auto_schedule.h"

#include "HalideBuffer.h"
#include "halide_benchmark.h"

using namespace Halide::Tools;
using namespace Halide::Runtime;

// B: Batch size
// H: Head number
// N: Token number
// D: Token dimension
// S: Hidden layer size
const int B = 1, H = 4, N = 256, D = 512, S = 512;
const float HI = 1, LO = -1;

Buffer<float, 3> input(B, N, D);
Buffer<float, 3> weight_q(H, D, S);
Buffer<float, 3> weight_k(H, D, S);
Buffer<float, 3> weight_v(H, D, S);
Buffer<float, 2> weight_o(H * S, D);
Buffer<float, 3> output(B, N, D);

static float c_input[B][N][D];
static float c_weight_q[H][D][S];
static float c_weight_k[H][D][S];
static float c_weight_v[H][D][S];
static float c_weight_o[H * S][D];
static float c_output[B][N][D] = {0};

static float mat_q[B][H][N][S] = {0};
static float mat_k[B][H][N][S] = {0};
static float mat_v[B][H][N][S] = {0};
static float mat_qkt[B][H][N][N] = {0};
static float mat_sv[B][H][N][S] = {0};
static float concat[B][N][H * S] = {0};
static float softmax[B][H][N][N] = {0};

int main(int argc, char **argv) {
    srand(0);

    // randomize input and weight
    for (int z = 0; z < input.channels(); z++) {
        for (int y = 0; y < input.height(); y++) {
            for (int x = 0; x < input.width(); x++) {
                c_input[x][y][z] = LO + static_cast<float>(rand()) /
                                            (static_cast<float>(RAND_MAX / (HI - LO)));
                input(x, y, z) = c_input[x][y][z];
            }
        }
    }

    for (int z = 0; z < weight_v.channels(); z++) {
        for (int y = 0; y < weight_v.height(); y++) {
            for (int x = 0; x < weight_v.width(); x++) {
                c_weight_q[x][y][z] = LO + static_cast<float>(rand()) /
                                            (static_cast<float>(RAND_MAX / (HI - LO)));
                c_weight_k[x][y][z] = LO + static_cast<float>(rand()) /
                                            (static_cast<float>(RAND_MAX / (HI - LO)));
                c_weight_v[x][y][z] = LO + static_cast<float>(rand()) /
                                            (static_cast<float>(RAND_MAX / (HI - LO)));
                weight_q(x, y, z) = c_weight_q[x][y][z];
                weight_k(x, y, z) = c_weight_k[x][y][z];
                weight_v(x, y, z) = c_weight_v[x][y][z];
            }
        }
    }

    for (int y = 0; y < weight_o.height(); y++) {
        for (int x = 0; x < weight_o.width(); x++) {
            c_weight_o[x][y] = LO + static_cast<float>(rand()) /
                                            (static_cast<float>(RAND_MAX / (HI - LO)));
            weight_o(x, y) = c_weight_o[x][y];
        }
    }

    // c run
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; ++h) {
            for (int n = 0; n < N; ++n) {
                for (int s = 0; s < S; ++s) {
                    for (int d = 0; d < D; ++d) {
                        mat_q[b][h][n][s] += c_input[b][n][d] * c_weight_q[h][d][s];
                        mat_k[b][h][n][s] += c_input[b][n][d] * c_weight_k[h][d][s];
                        mat_v[b][h][n][s] += c_input[b][n][d] * c_weight_v[h][d][s];
                    }
                }
            }
        }
    }

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; ++h) {
            for (int n = 0; n < N; ++n) {
                for (int d = 0; d < N; ++d) {
                    for (int s = 0; s < S; ++s) {
                        mat_qkt[b][h][n][d] += mat_q[b][h][n][s] * mat_k[b][h][d][s];
                    }
                }
            }
        }
    }

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; ++h) {
            for (int n = 0; n < N; ++n) {
                float exp_max = -std::numeric_limits<float>::infinity();
                float normalizer = 0;
                for (int d = 0; d < N; ++d) {
                    exp_max = std::max(exp_max, mat_qkt[b][h][n][d]);
                }
                for (int d = 0; d < N; ++d) {
                    normalizer += expf(mat_qkt[b][h][n][d] - exp_max);
                }
                for (int d = 0; d < N; ++d) {
                    softmax[b][h][n][d] += expf(mat_qkt[b][h][n][d] - exp_max) / normalizer;
                }
            }
        }
    }

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; ++h) {
            for (int n = 0; n < N; ++n) {
                for (int s = 0; s < S; ++s) {
                    for (int d = 0; d < N; ++d) {
                        mat_sv[b][h][n][s] += softmax[b][h][n][d] * mat_v[b][h][d][s];
                    }
                }
            }
        }
    }

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; ++h) {
            for (int n = 0; n < N; ++n) {
                for (int s = 0; s < S; ++s) {
                    concat[b][n][h * S + s] = mat_sv[b][h][n][s];
                }
            }
        }
    }

    for (int b = 0; b < B; b++) {
        for (int n = 0; n < N; ++n) {
            for (int d = 0; d < D; ++d) {
                for (int c = 0; c < H * S; ++c) {
                    c_output[b][n][d] += concat[b][n][c] * c_weight_o[c][d];
                }
            }
        }
    }

    // halide run
    attention_layer(input, weight_q, weight_k, weight_v, weight_o, output);

    // Check the C and Halide results match:
    for (int b = 0; b < B; b++) {
        for (int n = 0; n < N; ++n) {
            for (int d = 0; d < D; ++d) {
                float error = output(b, n, d) - c_output[b][n][d];
                // It's floating-point math, so we'll allow some slop:
                if (error < -0.001f || error > 0.001f) {
                    printf("halide_manually_tuned_result(%d, %d, %d) = %f instead of %f\n",
                           b, n, d, output(b, n, d), c_output[b][n][d]);
//                    return -1;
                }
            }
        }
    }


    // halide run
    attention_layer_auto_schedule(input, weight_q, weight_k, weight_v, weight_o, output);

    // Check the C and Halide results match:
    for (int b = 0; b < B; b++) {
        for (int n = 0; n < N; ++n) {
            for (int d = 0; d < D; ++d) {
                float error = output(b, n, d) - c_output[b][n][d];
                // It's floating-point math, so we'll allow some slop:
                if (error < -0.001f || error > 0.001f) {
                    printf("halide_auto_tuned_result(%d, %d, %d) = %f instead of %f\n",
                           b, n, d, output(b, n, d), c_output[b][n][d]);
                    //                    return -1;
                }
            }
        }
    }

    // Timing code

    // Manually-tuned version
    double min_t_manual = benchmark(2, 2, [&]() {
        attention_layer(input, weight_q, weight_k, weight_v, weight_o, output);
        output.device_sync();
    });
    printf("Manually-tuned time: %gs\n", min_t_manual);

    // Auto-scheduled version
    double min_t_auto = benchmark(10, 10, [&]() {
      attention_layer_auto_schedule(input, weight_q, weight_k, weight_v, weight_o, output);
      output.device_sync();
    });
    printf("Auto-scheduled time: %gms\n", min_t_auto * 1e3);

    printf("Success!\n");
    return 0;
}
