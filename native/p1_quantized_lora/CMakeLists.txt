cmake_minimum_required(VERSION 3.18)
project(xload_p1_quantized_lora CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Standalone on purpose: this CMakeLists builds with the HOST compiler, not
# the Android NDK, so Stage 1's numeric correctness can be proven (and
# re-run in CI) without needing an NDK toolchain or a device at all --
# wiring this into app's Android/JNI build is separate follow-up work,
# sequenced after the math itself is verified (see ../../docs/
# XLOAD_P1_STATUS.md).
add_executable(test_quantized_linear_lora tests/test_quantized_linear_lora.cpp)
target_include_directories(test_quantized_linear_lora PRIVATE include)

add_executable(test_half_to_float tests/test_half_to_float.cpp)
target_include_directories(test_half_to_float PRIVATE include)

enable_testing()
add_test(
  NAME quantized_linear_lora_fixtures
  COMMAND test_quantized_linear_lora
          "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/case_single_block.txt"
          "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/case_multi_block.txt"
)
add_test(NAME half_to_float COMMAND test_half_to_float)
