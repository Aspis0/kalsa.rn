# Standalone host test for the GGML_HEXAGON_VCORNER parser. Not wired into the
# main ggml build; configure and run on the build host:
#   cmake -S <this dir> -B <build> && cmake --build <build> && ctest --test-dir <build>
cmake_minimum_required(VERSION 3.16)
project(ggml-hexagon-lab-tests C)

add_executable(test-htp-vcorner test-htp-vcorner.c)
target_include_directories(test-htp-vcorner PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

enable_testing()
add_test(NAME htp-vcorner COMMAND test-htp-vcorner)
