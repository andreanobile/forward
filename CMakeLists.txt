cmake_minimum_required (VERSION 2.6)
project (nnlib_test)
# The version number.
set (nnlib_test_VERSION_MAJOR 1)
set (nnlib_test_VERSION_MINOR 0)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mfma -std=c++11 -fsanitize=address")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address")
#set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} -fsanitize=address")

set ( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -pthread")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -mavx -mfma -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -mavx -mfma -std=c++14")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx -mfma -std=c++11")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -g -mavx -mfma -Wall")

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/nnlib_test_config.h.in"
  "${PROJECT_BINARY_DIR}/nnlib_test_config.h"
  )

# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
include_directories("${PROJECT_BINARY_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/nnlib")
include_directories("${PROJECT_SOURCE_DIR}/blas")
include_directories("${HOME}/openblas/include")
add_subdirectory(nnlib)
add_subdirectory(blas)

# add the executable
add_executable(nnlib_test nnlib_test.cpp)
#target_link_libraries(nnlib_test nnlib /home/andrea/openblas/lib/libopenblas.a)
target_link_libraries(nnlib_test nnlib blas libturbojpeg.a libgd.a)
