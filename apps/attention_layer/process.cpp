#include <chrono>
#include <cstdio>

#include "attention_layer.h"

#include "HalideBuffer.h"
#include "halide_benchmark.h"

using namespace Halide::Tools;
using namespace Halide::Runtime;

int main(int argc, char **argv) {
    // B: Batch size
    // H: Head number
    // N: Token number
    // D: Token dimension
    // S: Hidden layer size
    const int B = 1, H = 1, N = 512, D = 768, S = 512;

    Buffer<float, 3> input(B, N, D);
    Buffer<float, 3> weight_q(H, D, S);
    Buffer<float, 3> weight_k(H, D, S);
    Buffer<float, 3> weight_v(H, D, S);
    Buffer<float, 2> weight_o(H * S, D);

    for (int z = 0; z < input.channels(); z++) {
        for (int y = 0; y < input.height(); y++) {
            for (int x = 0; x < input.width(); x++) {
                input(x, y, z) = rand();
            }
        }
    }

    for (int z = 0; z < weight_v.channels(); z++) {
        for (int y = 0; y < weight_v.height(); y++) {
            for (int x = 0; x < weight_v.width(); x++) {
                weight_q(x, y, z) = rand();
                weight_k(x, y, z) = rand();
                weight_v(x, y, z) = rand();
            }
        }
    }

    for (int y = 0; y < weight_k.height(); y++) {
        for (int x = 0; x < weight_k.width(); x++) {
            weight_o(x, y) = rand();
        }
    }

    Buffer<float, 3> output(B, N, D);

    attention_layer(input, weight_q, weight_k, weight_v, weight_o, output);

    // Timing code

    // Manually-tuned version
    double min_t_manual = benchmark(10, 10, [&]() {
      attention_layer(input, weight_q, weight_k, weight_v, weight_o, output);
      output.device_sync();
    });
    printf("Manually-tuned time: %gs\n", min_t_manual);

    printf("Success!\n");
    return 0;
}
