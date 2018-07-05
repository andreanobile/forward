project(assembler C ASM)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx -mfma")
add_library(blas matmat.cpp sgemm_itcopy_16.c sgemm_oncopy_4.c sgemm_kernel_16x4_haswell_preprocessed.S blas_memory.c)

