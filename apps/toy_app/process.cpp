#include <chrono>
#include <cmath>
#include <cstdio>

#include "toy_app.h"

#include "HalideBuffer.h"
#include "halide_benchmark.h"

using namespace Halide::Tools;
using namespace Halide::Runtime;

// N: Token number
// D: Token dimension
// S: Hidden layer size
const int N = 512, D = 64, S = 256;
const float HI = 1, LO = -1;

Buffer<float, 2> input(N, D);
Buffer<float, 2> weight_q(D, S);
Buffer<float, 2> weight_k(D, S);
Buffer<float, 2> output(N, N);

static float c_input[N][D];
static float c_weight_q[D][S];
static float c_weight_k[D][S];

int main(int argc, char **argv) {
    srand(0);

    // randomize input and weight
    for (int y = 0; y < input.height(); y++) {
        for (int x = 0; x < input.width(); x++) {
            c_input[x][y] = LO + static_cast<float>(rand()) /
                                     (static_cast<float>(RAND_MAX / (HI - LO)));
            input(x, y) = c_input[x][y];
        }
    }

    for (int y = 0; y < weight_q.height(); y++) {
        for (int x = 0; x < weight_q.width(); x++) {
            c_weight_q[x][y] = LO + static_cast<float>(rand()) /
                                        (static_cast<float>(RAND_MAX / (HI - LO)));
            c_weight_k[x][y] = LO + static_cast<float>(rand()) /
                                        (static_cast<float>(RAND_MAX / (HI - LO)));
            weight_q(x, y) = c_weight_q[x][y];
            weight_k(x, y) = c_weight_k[x][y];
        }
    }

    // halide run
    toy_app(input, weight_q, weight_k, output);

    // Timing code

    // Manually-tuned version
    double min_t_manual = benchmark(2, 2, [&]() {
        toy_app(input, weight_q, weight_k, output);
        output.device_sync();
    });
    printf("Manually-tuned time: %gs\n", min_t_manual);

    printf("Success!\n");
    return 0;
}
