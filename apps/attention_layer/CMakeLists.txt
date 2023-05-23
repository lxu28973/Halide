cmake_minimum_required(VERSION 3.22)
project(attention_layer)

enable_testing()

# Set up language settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Find Halide
find_package(Halide REQUIRED)

# Generator
add_halide_generator(attention_layer.generator SOURCES attention_layer_generator.cpp)

add_halide_library(attention_layer FROM attention_layer.generator
                   STMT attention.stmt
                   STMT_HTML attention.html
                   COMPILER_LOG attention.log)
add_halide_library(attention_layer_auto_schedule FROM attention_layer.generator
                   GENERATOR attention_layer
                   AUTOSCHEDULER Halide::Adams2019
                   SCHEDULE attention_auto.txt)

# Main executable
add_executable(attention_layer_process process.cpp)
target_link_libraries(attention_layer_process
                      PRIVATE
                      Halide::ImageIO
                      attention_layer
                      attention_layer_auto_schedule)

# Test that the app actually works!
add_test(NAME attention_layer_process COMMAND attention_layer_process)
set_tests_properties(attention_layer_process PROPERTIES
                     LABELS attention_layer
                     PASS_REGULAR_EXPRESSION "Success!"
                     SKIP_REGULAR_EXPRESSION "\\[SKIP\\]")
