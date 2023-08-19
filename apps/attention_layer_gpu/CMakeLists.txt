cmake_minimum_required(VERSION 3.22)
project(attention_layer_gpu)

enable_testing()

# Set up language settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Find Halide
find_package(Halide REQUIRED)

# Generator
add_halide_generator(attention_layer_gpu.generator SOURCES attention_layer_gpu_generator.cpp)

add_halide_library(attention_layer_gpu FROM attention_layer_gpu.generator
                    TARGETS host
                    FEATURES metal
                   STMT attention_gpu.stmt
                   STMT_HTML attention_gpu.html
                   COMPILER_LOG attention_gpu.log)

# Main executable
add_executable(attention_layer_gpu_process process.cpp)
target_link_libraries(attention_layer_gpu_process
                      PRIVATE
                      Halide::ImageIO
                      attention_layer_gpu
                      )

# Test that the app actually works!
add_test(NAME attention_layer_gpu_process COMMAND attention_layer_gpu_process)
set_tests_properties(attention_layer_gpu_process PROPERTIES
                     LABELS attention_layer_gpu
                     PASS_REGULAR_EXPRESSION "Success!"
                     SKIP_REGULAR_EXPRESSION "\\[SKIP\\]")
