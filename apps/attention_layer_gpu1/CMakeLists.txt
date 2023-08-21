cmake_minimum_required(VERSION 3.22)
project(attention_layer_gpu1)

enable_testing()

# Set up language settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Find Halide
find_package(Halide REQUIRED)

# Generator
add_halide_generator(attention_layer_gpu1.generator SOURCES attention_layer_gpu_generator.cpp)

add_halide_library(attention_layer_gpu1 FROM attention_layer_gpu1.generator
                    TARGETS host
                    FEATURES cuda cuda_capability_86
                   STMT attention_gpu.stmt
                   STMT_HTML attention_gpu.html
                   COMPILER_LOG attention_gpu.log)

# Main executable
add_executable(attention_layer_gpu_process1 process.cpp)
target_link_libraries(attention_layer_gpu_process1
                      PRIVATE
                      Halide::ImageIO
                      attention_layer_gpu1
                      )

# Test that the app actually works!
add_test(NAME attention_layer_gpu_process1 COMMAND attention_layer_gpu_process1)
set_tests_properties(attention_layer_gpu_process1 PROPERTIES
                     LABELS attention_layer_gpu1
                     PASS_REGULAR_EXPRESSION "Success!"
                     SKIP_REGULAR_EXPRESSION "\\[SKIP\\]")
