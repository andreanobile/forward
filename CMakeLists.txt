cmake_minimum_required (VERSION 2.6)
project (nnlib_test)
# The version number.
set (nnlib_test_VERSION_MAJOR 1)
set (nnlib_test_VERSION_MINOR 0)

set ( CMAKE_EXE_LINKER_FLAGS_RELEASE  "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -pthread")
set ( CMAKE_EXE_LINKER_FLAGS_DEBUG  "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=address -fsanitize=undefined -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}  -mavx -mfma -std=c++14 -Wall -O0 -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined -D__FORTIFY_SOURCE=2 -rdynamic -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -mavx -mfma -std=c++14 -Wall")


set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -mavx -mfma -Wall -O0 -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined -D__FORTIFY_SOURCE=2 -rdynamic -DDEBUG")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O2 -g -mavx -mfma -Wall")

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/nnlib_test_config.h.in"
  "${PROJECT_BINARY_DIR}/nnlib_test_config.h"
  )


include_directories("${PROJECT_BINARY_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/nnlib")
include_directories("${PROJECT_SOURCE_DIR}/blas")
include_directories("${HOME}/openblas/include")
add_subdirectory(nnlib)
add_subdirectory(blas)

# add the executable
add_executable(nnlib_test nnlib_test.cpp)

target_link_libraries(nnlib_test nnlib blas libturbojpeg.a libgd.a)
