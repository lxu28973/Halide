cmake_minimum_required(VERSION 3.22)
project(toy_app)

enable_testing()

# Set up language settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Find Halide
find_package(Halide REQUIRED)

# Generator
add_halide_generator(toy_app.generator SOURCES toy_app.cpp)

# Filters
add_halide_library(toy_app FROM toy_app.generator
                    TARGETS host
                    FEATURES metal
                   STMT toy_app.stmt
                   STMT_HTML toy_app.html
                   COMPILER_LOG toy_app.log)

# Main executable
add_executable(toy_app_process process.cpp)
target_link_libraries(toy_app_process
                      PRIVATE
                      Halide::ImageIO
                      toy_app)

# Test that the app actually works!
add_test(NAME toy_app_process COMMAND toy_app_process)
set_tests_properties(toy_app_process PROPERTIES
                     LABELS toy_app
                     PASS_REGULAR_EXPRESSION "Success!"
                     SKIP_REGULAR_EXPRESSION "\\[SKIP\\]")
