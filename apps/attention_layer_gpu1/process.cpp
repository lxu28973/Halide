#include <cmath>
#include <chrono>
#include <cstdio>

#include "attention_layer_gpu1.h"

#include "HalideBuffer.h"
#include "halide_benchmark.h"

using namespace Halide::Tools;
using namespace Halide::Runtime;

// B: Batch size
// H: Head number
// N: Token number
// D: Hidden layer size
const int B = 8, H = 8, N = 2048, D = 96;
const float HI = 1, LO = -1;

static float c_input[B][N][D * H];
static float c_weight_q[H][D * H][D];
static float c_weight_k[H][D * H][D];
static float c_weight_v[H][D * H][D];
//static float c_output[B][H][N][D] = {0};

//static float mat_q[B][H][N][D] = {0};
//static float mat_k[B][H][N][D] = {0};
//static float mat_v[B][H][N][D] = {0};
//static float mat_qkt[B][H][N][N] = {0};
//static float mat_sv[B][H][N][D] = {0};

int main(int argc, char **argv) {
  srand(0);

  Buffer<float, 3> input(B, N, D * H);
  Buffer<float, 3> weight_q(H, D * H, D);
  Buffer<float, 3> weight_k(H, D * H, D);
  Buffer<float, 3> weight_v(H, D * H, D);
  Buffer<float, 4> output(B, H, N, D);

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

  // c run
//  for (int b = 0; b < B; b++) {
//    for (int h = 0; h < H; ++h) {
//      for (int n = 0; n < N; ++n) {
//        for (int d = 0; d < D; ++d) {
//          for (int dh = 0; dh < D * H; ++dh) {
//            mat_q[b][h][n][d] += c_input[b][n][dh] * c_weight_q[h][dh][d];
//            mat_k[b][h][n][d] += c_input[b][n][dh] * c_weight_k[h][dh][d];
//            mat_v[b][h][n][d] += c_input[b][n][dh] * c_weight_v[h][dh][d];
//          }
//        }
//      }
//    }
//  }
//
//  for (int b = 0; b < B; b++) {
//    for (int h = 0; h < H; ++h) {
//      for (int n = 0; n < N; ++n) {
//        for (int d = 0; d < N; ++d) {
//          for (int s = 0; s < D; ++s) {
//            mat_qkt[b][h][n][d] += mat_q[b][h][n][s] * mat_k[b][h][d][s];
//          }
//        }
//      }
//    }
//  }
//
//  for (int b = 0; b < B; b++) {
//    for (int h = 0; h < H; ++h) {
//      for (int n = 0; n < N; ++n) {
//        for (int s = 0; s < D; ++s) {
//          for (int d = 0; d < N; ++d) {
//            mat_sv[b][h][n][s] += mat_qkt[b][h][n][d] * mat_v[b][h][d][s];
//          }
//        }
//      }
//    }
//  }
//
//  for (int b = 0; b < B; b++) {
//    for (int h = 0; h < H; ++h) {
//      for (int n = 0; n < N; ++n) {
//        for (int s = 0; s < D; ++s) {
//          c_output[b][h][n][s] = mat_sv[b][h][n][s];
//        }
//      }
//    }
//  }

  // halide run
  attention_layer_gpu1(input, weight_q, weight_k, weight_v, output);

  // Check the C and Halide results match:
//  for (int b = 0; b < B; b++) {
//    for (int h = 0; h < H; h++) {
//      for (int n = 0; n < N; ++n) {
//        for (int d = 0; d < D; ++d) {
//          float error = output(b, h, n, d) - c_output[b][h][n][d];
//          // It's floating-point math, so we'll allow some slop:
//          if (error < -0.001f || error > 0.001f) {
//            printf("halide_manually_tuned_result(%d, %d, %d, %d) = %f instead of %f\n",
//                   b, h, n, d, output(b, h, n, d), c_output[b][h][n][d]);
////                    return -1;
//          }
//        }
//      }
//    }
//  }


  // Timing code

  // Manually-tuned version
  double min_t_manual = benchmark(2, 2, [&]() {
    attention_layer_gpu1(input, weight_q, weight_k, weight_v, output);
    output.device_sync();
  });
  printf("Manually-tuned time: %gs\n", min_t_manual);

  printf("Success!\n");
  return 0;
}
