/*********************************************************************************
Copyright (c) 2013, The OpenBLAS Project
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in
the documentation and/or other materials provided with the
distribution.
3. Neither the name of the OpenBLAS project nor the names of
its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE OPENBLAS PROJECT OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********************************************************************************/

/*********************************************************************
* 2014/07/28 Saar
*        BLASTEST               : OK
*        CTEST                  : OK
*        TEST                   : OK
*
* 2013/10/28 Saar
* Parameter:
*	SGEMM_DEFAULT_UNROLL_N	4
*	SGEMM_DEFAULT_UNROLL_M	16
*	SGEMM_DEFAULT_P		768
*	SGEMM_DEFAULT_Q		384
*	A_PR1			512
*	B_PR1			512
*	
* 
* 2014/07/28 Saar
* Performance at 9216x9216x9216:
*       1 thread:      102 GFLOPS       (SANDYBRIDGE:  59)      (MKL:   83)
*       2 threads:     195 GFLOPS       (SANDYBRIDGE: 116)      (MKL:  155)
*       3 threads:     281 GFLOPS       (SANDYBRIDGE: 165)      (MKL:  230)
*       4 threads:     366 GFLOPS       (SANDYBRIDGE: 223)      (MKL:  267)
*
*********************************************************************/

# 1 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/usr/include/stdc-predef.h" 1 3 4
# 1 "<command-line>" 2
# 1 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
# 54 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
# 1 "../common.h" 1
# 62 "../common.h"
# 1 "../config.h" 1
# 63 "../common.h" 2
# 392 "../common.h"
# 1 "../common_x86_64.h" 1
# 393 "../common.h" 2
# 487 "../common.h"
# 1 "../common_linux.h" 1
# 488 "../common.h" 2
# 498 "../common.h"
# 1 "../param.h" 1
# 499 "../common.h" 2
# 1 "../common_param.h" 1
# 500 "../common.h" 2
# 749 "../common.h"
# 1 "../common_interface.h" 1
# 750 "../common.h" 2



# 1 "../common_macro.h" 1
# 42 "../common_macro.h"
# 1 "../common_s.h" 1
# 43 "../common_macro.h" 2
# 1 "../common_d.h" 1
# 44 "../common_macro.h" 2
# 1 "../common_q.h" 1
# 45 "../common_macro.h" 2

# 1 "../common_c.h" 1
# 47 "../common_macro.h" 2
# 1 "../common_z.h" 1
# 48 "../common_macro.h" 2
# 1 "../common_x.h" 1
# 49 "../common_macro.h" 2
# 754 "../common.h" 2
# 1 "../common_level1.h" 1
# 755 "../common.h" 2
# 1 "../common_level2.h" 1
# 756 "../common.h" 2
# 1 "../common_level3.h" 1
# 757 "../common.h" 2
# 1 "../common_lapack.h" 1
# 758 "../common.h" 2
# 55 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S" 2
# 158 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
.macro KERNEL16x6_SUB
 vmovups -16 * 4(%rdi), %ymm0
 vmovups -8 * 4(%rdi), %ymm1
 vbroadcastss -4 * 4(%rsi), %ymm2
 vbroadcastss -3 * 4(%rsi), %ymm3
 prefetcht0 512(%rdi)

 vfmadd231ps %ymm2,%ymm0,%ymm4
 vfmadd231ps %ymm2,%ymm1,%ymm5
 vfmadd231ps %ymm3,%ymm0,%ymm6
 vfmadd231ps %ymm3,%ymm1,%ymm7

 vbroadcastss -2 * 4(%rsi), %ymm2
 vbroadcastss -1 * 4(%rsi), %ymm3
 vfmadd231ps %ymm2,%ymm0,%ymm8
 vfmadd231ps %ymm2,%ymm1,%ymm9
 vfmadd231ps %ymm3,%ymm0,%ymm10
 vfmadd231ps %ymm3,%ymm1,%ymm11

 vbroadcastss 0 * 4(%rsi), %ymm2
 vbroadcastss 1 * 4(%rsi), %ymm3
 vfmadd231ps %ymm2,%ymm0,%ymm12
 vfmadd231ps %ymm2,%ymm1,%ymm13
 vfmadd231ps %ymm3,%ymm0,%ymm14
 vfmadd231ps %ymm3,%ymm1,%ymm15

 addq $ 6*4, %rsi
 addq $ 16*4, %rdi
 decq %rax
.endm

.macro SAVE16x6

 vbroadcastss 48(%rsp), %ymm0

 vmulps %ymm0 , %ymm4 , %ymm4
 vmulps %ymm0 , %ymm5 , %ymm5
 vmulps %ymm0 , %ymm6 , %ymm6
 vmulps %ymm0 , %ymm7 , %ymm7
 vmulps %ymm0 , %ymm8 , %ymm8
 vmulps %ymm0 , %ymm9 , %ymm9
 vmulps %ymm0 , %ymm10, %ymm10
 vmulps %ymm0 , %ymm11, %ymm11
 vmulps %ymm0 , %ymm12, %ymm12
 vmulps %ymm0 , %ymm13, %ymm13
 vmulps %ymm0 , %ymm14, %ymm14
 vmulps %ymm0 , %ymm15, %ymm15




 vaddps (%r15), %ymm4,%ymm4
 vaddps 8 * 4(%r15), %ymm5,%ymm5

 vaddps (%r15, %r10), %ymm6,%ymm6
 vaddps 8 * 4(%r15, %r10), %ymm7,%ymm7

 vaddps (%r15, %r10,2), %ymm8,%ymm8
 vaddps 8 * 4(%r15, %r10,2), %ymm9,%ymm9

 vaddps (%rdx), %ymm10,%ymm10
 vaddps 8 * 4(%rdx), %ymm11,%ymm11

 vaddps (%rdx, %r10), %ymm12,%ymm12
 vaddps 8 * 4(%rdx, %r10), %ymm13,%ymm13

 vaddps (%rdx, %r10,2), %ymm14,%ymm14
 vaddps 8 * 4(%rdx, %r10,2), %ymm15,%ymm15



 vmovups %ymm4 , (%r15)
 vmovups %ymm5 , 8 * 4(%r15)

 vmovups %ymm6 , (%r15, %r10)
 vmovups %ymm7 , 8 * 4(%r15, %r10)

 vmovups %ymm8 , (%r15, %r10,2)
 vmovups %ymm9 , 8 * 4(%r15, %r10,2)

 vmovups %ymm10, (%rdx)
 vmovups %ymm11, 8 * 4(%rdx)

 vmovups %ymm12, (%rdx, %r10)
 vmovups %ymm13, 8 * 4(%rdx, %r10)

 vmovups %ymm14, (%rdx, %r10,2)
 vmovups %ymm15, 8 * 4(%rdx, %r10,2)

.endm





.macro KERNEL8x6_SUB
 vmovups -16 * 4(%rdi), %ymm0
 vbroadcastss -4 * 4(%rsi), %ymm2
 vbroadcastss -3 * 4(%rsi), %ymm3

 vfmadd231ps %ymm2,%ymm0,%ymm4
 vfmadd231ps %ymm3,%ymm0,%ymm6

 vbroadcastss -2 * 4(%rsi), %ymm2
 vbroadcastss -1 * 4(%rsi), %ymm3
 vfmadd231ps %ymm2,%ymm0,%ymm8
 vfmadd231ps %ymm3,%ymm0,%ymm10

 vbroadcastss 0 * 4(%rsi), %ymm2
 vbroadcastss 1 * 4(%rsi), %ymm3
 vfmadd231ps %ymm2,%ymm0,%ymm12
 vfmadd231ps %ymm3,%ymm0,%ymm14

 addq $ 6*4, %rsi
 addq $ 8*4, %rdi
 decq %rax
.endm

.macro SAVE8x6

 vbroadcastss 48(%rsp), %ymm0

 vmulps %ymm0 , %ymm4 , %ymm4
 vmulps %ymm0 , %ymm6 , %ymm6
 vmulps %ymm0 , %ymm8 , %ymm8
 vmulps %ymm0 , %ymm10, %ymm10
 vmulps %ymm0 , %ymm12, %ymm12
 vmulps %ymm0 , %ymm14, %ymm14




 vaddps (%r15), %ymm4,%ymm4
 vaddps (%r15, %r10), %ymm6,%ymm6
 vaddps (%r15, %r10,2), %ymm8,%ymm8
 vaddps (%rdx), %ymm10,%ymm10
 vaddps (%rdx, %r10), %ymm12,%ymm12
 vaddps (%rdx, %r10,2), %ymm14,%ymm14



 vmovups %ymm4 , (%r15)
 vmovups %ymm6 , (%r15, %r10)
 vmovups %ymm8 , (%r15, %r10,2)
 vmovups %ymm10, (%rdx)
 vmovups %ymm12, (%rdx, %r10)
 vmovups %ymm14, (%rdx, %r10,2)

.endm





.macro KERNEL4x6_SUB
 vmovups -16 * 4(%rdi), %xmm0
 vbroadcastss -4 * 4(%rsi), %xmm2
 vbroadcastss -3 * 4(%rsi), %xmm3

 vfmadd231ps %xmm2,%xmm0,%xmm4
 vfmadd231ps %xmm3,%xmm0,%xmm6

 vbroadcastss -2 * 4(%rsi), %xmm2
 vbroadcastss -1 * 4(%rsi), %xmm3
 vfmadd231ps %xmm2,%xmm0,%xmm8
 vfmadd231ps %xmm3,%xmm0,%xmm10

 vbroadcastss 0 * 4(%rsi), %xmm2
 vbroadcastss 1 * 4(%rsi), %xmm3
 vfmadd231ps %xmm2,%xmm0,%xmm12
 vfmadd231ps %xmm3,%xmm0,%xmm14

 addq $ 6*4, %rsi
 addq $ 4*4, %rdi
 decq %rax
.endm

.macro SAVE4x6

 vbroadcastss 48(%rsp), %xmm0

 vmulps %xmm0 , %xmm4 , %xmm4
 vmulps %xmm0 , %xmm6 , %xmm6
 vmulps %xmm0 , %xmm8 , %xmm8
 vmulps %xmm0 , %xmm10, %xmm10
 vmulps %xmm0 , %xmm12, %xmm12
 vmulps %xmm0 , %xmm14, %xmm14




 vaddps (%r15), %xmm4,%xmm4
 vaddps (%r15, %r10), %xmm6,%xmm6
 vaddps (%r15, %r10,2), %xmm8,%xmm8
 vaddps (%rdx), %xmm10,%xmm10
 vaddps (%rdx, %r10), %xmm12,%xmm12
 vaddps (%rdx, %r10,2), %xmm14,%xmm14



 vmovups %xmm4 , (%r15)
 vmovups %xmm6 , (%r15, %r10)
 vmovups %xmm8 , (%r15, %r10,2)
 vmovups %xmm10, (%rdx)
 vmovups %xmm12, (%rdx, %r10)
 vmovups %xmm14, (%rdx, %r10,2)

.endm




.macro KERNEL2x6_SUB
 vmovss -16 * 4(%rdi), %xmm0
 vmovss -15 * 4(%rdi), %xmm1
 vmovss -4 * 4(%rsi), %xmm2
 vmovss -3 * 4(%rsi), %xmm3

 vfmadd231ss %xmm2,%xmm0,%xmm4
 vfmadd231ss %xmm2,%xmm1,%xmm5
 vfmadd231ss %xmm3,%xmm0,%xmm6
 vfmadd231ss %xmm3,%xmm1,%xmm7

 vmovss -2 * 4(%rsi), %xmm2
 vmovss -1 * 4(%rsi), %xmm3
 vfmadd231ss %xmm2,%xmm0,%xmm8
 vfmadd231ss %xmm2,%xmm1,%xmm9
 vfmadd231ss %xmm3,%xmm0,%xmm10
 vfmadd231ss %xmm3,%xmm1,%xmm11

 vmovss 0 * 4(%rsi), %xmm2
 vmovss 1 * 4(%rsi), %xmm3
 vfmadd231ss %xmm2,%xmm0,%xmm12
 vfmadd231ss %xmm2,%xmm1,%xmm13
 vfmadd231ss %xmm3,%xmm0,%xmm14
 vfmadd231ss %xmm3,%xmm1,%xmm15

 addq $ 6*4, %rsi
 addq $ 2*4, %rdi
 decq %rax
.endm

.macro SAVE2x6

 vmovss 48(%rsp), %xmm0

 vmulss %xmm0 , %xmm4 , %xmm4
 vmulss %xmm0 , %xmm5 , %xmm5
 vmulss %xmm0 , %xmm6 , %xmm6
 vmulss %xmm0 , %xmm7 , %xmm7
 vmulss %xmm0 , %xmm8 , %xmm8
 vmulss %xmm0 , %xmm9 , %xmm9
 vmulss %xmm0 , %xmm10, %xmm10
 vmulss %xmm0 , %xmm11, %xmm11
 vmulss %xmm0 , %xmm12, %xmm12
 vmulss %xmm0 , %xmm13, %xmm13
 vmulss %xmm0 , %xmm14, %xmm14
 vmulss %xmm0 , %xmm15, %xmm15




 vaddss (%r15), %xmm4,%xmm4
 vaddss 1 * 4(%r15), %xmm5,%xmm5

 vaddss (%r15, %r10), %xmm6,%xmm6
 vaddss 1 * 4(%r15, %r10), %xmm7,%xmm7

 vaddss (%r15, %r10,2), %xmm8,%xmm8
 vaddss 1 * 4(%r15, %r10,2), %xmm9,%xmm9

 vaddss (%rdx), %xmm10,%xmm10
 vaddss 1 * 4(%rdx), %xmm11,%xmm11

 vaddss (%rdx, %r10), %xmm12,%xmm12
 vaddss 1 * 4(%rdx, %r10), %xmm13,%xmm13

 vaddss (%rdx, %r10,2), %xmm14,%xmm14
 vaddss 1 * 4(%rdx, %r10,2), %xmm15,%xmm15



 vmovss %xmm4 , (%r15)
 vmovss %xmm5 , 1 * 4(%r15)

 vmovss %xmm6 , (%r15, %r10)
 vmovss %xmm7 , 1 * 4(%r15, %r10)

 vmovss %xmm8 , (%r15, %r10,2)
 vmovss %xmm9 , 1 * 4(%r15, %r10,2)

 vmovss %xmm10, (%rdx)
 vmovss %xmm11, 1 * 4(%rdx)

 vmovss %xmm12, (%rdx, %r10)
 vmovss %xmm13, 1 * 4(%rdx, %r10)

 vmovss %xmm14, (%rdx, %r10,2)
 vmovss %xmm15, 1 * 4(%rdx, %r10,2)

.endm




.macro KERNEL1x6_SUB
 vmovss -16 * 4(%rdi), %xmm0
 vmovss -4 * 4(%rsi), %xmm2
 vmovss -3 * 4(%rsi), %xmm3

 vfmadd231ss %xmm2,%xmm0,%xmm4
 vfmadd231ss %xmm3,%xmm0,%xmm6

 vmovss -2 * 4(%rsi), %xmm2
 vmovss -1 * 4(%rsi), %xmm3
 vfmadd231ss %xmm2,%xmm0,%xmm8
 vfmadd231ss %xmm3,%xmm0,%xmm10

 vmovss 0 * 4(%rsi), %xmm2
 vmovss 1 * 4(%rsi), %xmm3
 vfmadd231ss %xmm2,%xmm0,%xmm12
 vfmadd231ss %xmm3,%xmm0,%xmm14

 addq $ 6*4, %rsi
 addq $ 1*4, %rdi
 decq %rax
.endm

.macro SAVE1x6

 vmovss 48(%rsp), %xmm0

 vmulss %xmm0 , %xmm4 , %xmm4
 vmulss %xmm0 , %xmm6 , %xmm6
 vmulss %xmm0 , %xmm8 , %xmm8
 vmulss %xmm0 , %xmm10, %xmm10
 vmulss %xmm0 , %xmm12, %xmm12
 vmulss %xmm0 , %xmm14, %xmm14



 vaddss (%r15), %xmm4,%xmm4
 vaddss (%r15, %r10), %xmm6,%xmm6
 vaddss (%r15, %r10,2), %xmm8,%xmm8
 vaddss (%rdx), %xmm10,%xmm10
 vaddss (%rdx, %r10), %xmm12,%xmm12
 vaddss (%rdx, %r10,2), %xmm14,%xmm14



 vmovss %xmm4 , (%r15)
 vmovss %xmm6 , (%r15, %r10)
 vmovss %xmm8 , (%r15, %r10,2)
 vmovss %xmm10, (%rdx)
 vmovss %xmm12, (%rdx, %r10)
 vmovss %xmm14, (%rdx, %r10,2)

.endm
# 525 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
.macro KERNEL16x4_SUB
 vmovups -16 * 4(%rdi, %rax, 4), %ymm0
 vmovups -8 * 4(%rdi, %rax, 4), %ymm1
 vbroadcastss -4 * 4(%rsi, %rbp, 4), %ymm2
 vbroadcastss -3 * 4(%rsi, %rbp, 4), %ymm3
 vfmadd231ps %ymm2,%ymm0,%ymm4
 vfmadd231ps %ymm2,%ymm1,%ymm5
 vfmadd231ps %ymm3,%ymm0,%ymm6
 vfmadd231ps %ymm3,%ymm1,%ymm7
 vbroadcastss -2 * 4(%rsi, %rbp, 4), %ymm2
 vbroadcastss -1 * 4(%rsi, %rbp, 4), %ymm3
 vfmadd231ps %ymm2,%ymm0,%ymm8
 vfmadd231ps %ymm2,%ymm1,%ymm9
 vfmadd231ps %ymm3,%ymm0,%ymm10
 vfmadd231ps %ymm3,%ymm1,%ymm11
 addq $ 4 , %rbp
 addq $ 16, %rax
.endm

.macro SAVE16x4

 vbroadcastss 48(%rsp), %ymm0

 vmulps %ymm0 , %ymm4 , %ymm4
 vmulps %ymm0 , %ymm5 , %ymm5
 vmulps %ymm0 , %ymm6 , %ymm6
 vmulps %ymm0 , %ymm7 , %ymm7
 vmulps %ymm0 , %ymm8 , %ymm8
 vmulps %ymm0 , %ymm9 , %ymm9
 vmulps %ymm0 , %ymm10, %ymm10
 vmulps %ymm0 , %ymm11, %ymm11




 vaddps (%r15), %ymm4,%ymm4
 vaddps 8 * 4(%r15), %ymm5,%ymm5

 vaddps (%r15, %r10), %ymm6,%ymm6
 vaddps 8 * 4(%r15, %r10), %ymm7,%ymm7

 vaddps (%rdx), %ymm8,%ymm8
 vaddps 8 * 4(%rdx), %ymm9,%ymm9

 vaddps (%rdx, %r10), %ymm10,%ymm10
 vaddps 8 * 4(%rdx, %r10), %ymm11,%ymm11



 vmovups %ymm4 , (%r15)
 vmovups %ymm5 , 8 * 4(%r15)

 vmovups %ymm6 , (%r15, %r10)
 vmovups %ymm7 , 8 * 4(%r15, %r10)

 vmovups %ymm8 , (%rdx)
 vmovups %ymm9 , 8 * 4(%rdx)

 vmovups %ymm10, (%rdx, %r10)
 vmovups %ymm11, 8 * 4(%rdx, %r10)

 prefetcht0 64(%r15)
 prefetcht0 64(%r15, %r10)
 prefetcht0 64(%rdx)
 prefetcht0 64(%rdx, %r10)

.endm





.macro KERNEL8x4_SUB
 vmovups -16 * 4(%rdi, %rax, 4), %ymm0
 vbroadcastss -4 * 4(%rsi, %rbp, 4), %ymm2
 vbroadcastss -3 * 4(%rsi, %rbp, 4), %ymm3
 vfmadd231ps %ymm2,%ymm0,%ymm4
 vfmadd231ps %ymm3,%ymm0,%ymm6
 vbroadcastss -2 * 4(%rsi, %rbp, 4), %ymm2
 vbroadcastss -1 * 4(%rsi, %rbp, 4), %ymm3
 vfmadd231ps %ymm2,%ymm0,%ymm8
 vfmadd231ps %ymm3,%ymm0,%ymm10
 addq $ 4 , %rbp
 addq $ 8 , %rax
.endm

.macro SAVE8x4

 vbroadcastss 48(%rsp), %ymm0

 vmulps %ymm0 , %ymm4 , %ymm4
 vmulps %ymm0 , %ymm6 , %ymm6
 vmulps %ymm0 , %ymm8 , %ymm8
 vmulps %ymm0 , %ymm10, %ymm10




 vaddps (%r15), %ymm4,%ymm4
 vaddps (%r15, %r10), %ymm6,%ymm6
 vaddps (%rdx), %ymm8,%ymm8
 vaddps (%rdx, %r10), %ymm10,%ymm10



 vmovups %ymm4 , (%r15)
 vmovups %ymm6 , (%r15, %r10)
 vmovups %ymm8 , (%rdx)
 vmovups %ymm10, (%rdx, %r10)

.endm





.macro KERNEL4x4_SUB
 vmovups -16 * 4(%rdi, %rax, 4), %xmm0
 vbroadcastss -4 * 4(%rsi, %rbp, 4), %xmm2
 vbroadcastss -3 * 4(%rsi, %rbp, 4), %xmm3
 vfmadd231ps %xmm2,%xmm0,%xmm4
 vfmadd231ps %xmm3,%xmm0,%xmm6
 vbroadcastss -2 * 4(%rsi, %rbp, 4), %xmm2
 vbroadcastss -1 * 4(%rsi, %rbp, 4), %xmm3
 vfmadd231ps %xmm2,%xmm0,%xmm8
 vfmadd231ps %xmm3,%xmm0,%xmm10
 addq $ 4 , %rbp
 addq $ 4 , %rax
.endm

.macro SAVE4x4

 vbroadcastss 48(%rsp), %xmm0

 vmulps %xmm0 , %xmm4 , %xmm4
 vmulps %xmm0 , %xmm6 , %xmm6
 vmulps %xmm0 , %xmm8 , %xmm8
 vmulps %xmm0 , %xmm10, %xmm10




 vaddps (%r15), %xmm4,%xmm4
 vaddps (%r15, %r10), %xmm6,%xmm6
 vaddps (%rdx), %xmm8,%xmm8
 vaddps (%rdx, %r10), %xmm10,%xmm10



 vmovups %xmm4 , (%r15)
 vmovups %xmm6 , (%r15, %r10)
 vmovups %xmm8 , (%rdx)
 vmovups %xmm10, (%rdx, %r10)

.endm




.macro KERNEL2x4_SUB
 vmovss -16 * 4(%rdi, %rax, 4), %xmm0
 vmovss -15 * 4(%rdi, %rax, 4), %xmm1
 vmovss -4 * 4(%rsi, %rbp, 4), %xmm2
 vmovss -3 * 4(%rsi, %rbp, 4), %xmm3
 vfmadd231ss %xmm2,%xmm0,%xmm4
 vfmadd231ss %xmm2,%xmm1,%xmm5
 vfmadd231ss %xmm3,%xmm0,%xmm6
 vfmadd231ss %xmm3,%xmm1,%xmm7
 vmovss -2 * 4(%rsi, %rbp, 4), %xmm2
 vmovss -1 * 4(%rsi, %rbp, 4), %xmm3
 vfmadd231ss %xmm2,%xmm0,%xmm8
 vfmadd231ss %xmm2,%xmm1,%xmm9
 vfmadd231ss %xmm3,%xmm0,%xmm10
 vfmadd231ss %xmm3,%xmm1,%xmm11
 addq $ 4 , %rbp
 addq $ 2, %rax
.endm

.macro SAVE2x4

 vmovss 48(%rsp), %xmm0

 vmulss %xmm0 , %xmm4 , %xmm4
 vmulss %xmm0 , %xmm5 , %xmm5
 vmulss %xmm0 , %xmm6 , %xmm6
 vmulss %xmm0 , %xmm7 , %xmm7
 vmulss %xmm0 , %xmm8 , %xmm8
 vmulss %xmm0 , %xmm9 , %xmm9
 vmulss %xmm0 , %xmm10, %xmm10
 vmulss %xmm0 , %xmm11, %xmm11




 vaddss (%r15), %xmm4,%xmm4
 vaddss 1 * 4(%r15), %xmm5,%xmm5

 vaddss (%r15, %r10), %xmm6,%xmm6
 vaddss 1 * 4(%r15, %r10), %xmm7,%xmm7

 vaddss (%rdx), %xmm8,%xmm8
 vaddss 1 * 4(%rdx), %xmm9,%xmm9

 vaddss (%rdx, %r10), %xmm10,%xmm10
 vaddss 1 * 4(%rdx, %r10), %xmm11,%xmm11



 vmovss %xmm4 , (%r15)
 vmovss %xmm5 , 1 * 4(%r15)

 vmovss %xmm6 , (%r15, %r10)
 vmovss %xmm7 , 1 * 4(%r15, %r10)

 vmovss %xmm8 , (%rdx)
 vmovss %xmm9 , 1 * 4(%rdx)

 vmovss %xmm10, (%rdx, %r10)
 vmovss %xmm11, 1 * 4(%rdx, %r10)

.endm




.macro KERNEL1x4_SUB
 vmovss -16 * 4(%rdi, %rax, 4), %xmm0
 vmovss -4 * 4(%rsi, %rbp, 4), %xmm2
 vmovss -3 * 4(%rsi, %rbp, 4), %xmm3
 vfmadd231ss %xmm2,%xmm0,%xmm4
 vfmadd231ss %xmm3,%xmm0,%xmm6
 vmovss -2 * 4(%rsi, %rbp, 4), %xmm2
 vmovss -1 * 4(%rsi, %rbp, 4), %xmm3
 vfmadd231ss %xmm2,%xmm0,%xmm8
 vfmadd231ss %xmm3,%xmm0,%xmm10
 addq $ 4 , %rbp
 addq $ 1, %rax
.endm

.macro SAVE1x4

 vmovss 48(%rsp), %xmm0

 vmulss %xmm0 , %xmm4 , %xmm4
 vmulss %xmm0 , %xmm6 , %xmm6
 vmulss %xmm0 , %xmm8 , %xmm8
 vmulss %xmm0 , %xmm10, %xmm10




 vaddss (%r15), %xmm4,%xmm4
 vaddss (%r15, %r10), %xmm6,%xmm6
 vaddss (%rdx), %xmm8,%xmm8
 vaddss (%rdx, %r10), %xmm10,%xmm10



 vmovss %xmm4 , (%r15)
 vmovss %xmm6 , (%r15, %r10)
 vmovss %xmm8 , (%rdx)
 vmovss %xmm10, (%rdx, %r10)

.endm
# 797 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
.macro KERNEL16x2_SUB
 vmovups -16 * 4(%rdi, %rax, 4), %ymm0
 vmovups -8 * 4(%rdi, %rax, 4), %ymm1
 vbroadcastss -4 * 4(%rsi, %rbp, 4), %ymm2
 vbroadcastss -3 * 4(%rsi, %rbp, 4), %ymm3
 vfmadd231ps %ymm2,%ymm0,%ymm4
 vfmadd231ps %ymm2,%ymm1,%ymm5
 vfmadd231ps %ymm3,%ymm0,%ymm6
 vfmadd231ps %ymm3,%ymm1,%ymm7
 addq $ 2 , %rbp
 addq $ 16, %rax
.endm

.macro SAVE16x2

 vbroadcastss 48(%rsp), %ymm0

 vmulps %ymm0 , %ymm4 , %ymm4
 vmulps %ymm0 , %ymm5 , %ymm5
 vmulps %ymm0 , %ymm6 , %ymm6
 vmulps %ymm0 , %ymm7 , %ymm7




 vaddps (%r15), %ymm4,%ymm4
 vaddps 8 * 4(%r15), %ymm5,%ymm5

 vaddps (%r15, %r10), %ymm6,%ymm6
 vaddps 8 * 4(%r15, %r10), %ymm7,%ymm7



 vmovups %ymm4 , (%r15)
 vmovups %ymm5 , 8 * 4(%r15)

 vmovups %ymm6 , (%r15, %r10)
 vmovups %ymm7 , 8 * 4(%r15, %r10)

.endm





.macro KERNEL8x2_SUB
 vmovups -16 * 4(%rdi, %rax, 4), %ymm0
 vbroadcastss -4 * 4(%rsi, %rbp, 4), %ymm2
 vbroadcastss -3 * 4(%rsi, %rbp, 4), %ymm3
 vfmadd231ps %ymm2,%ymm0,%ymm4
 vfmadd231ps %ymm3,%ymm0,%ymm6
 addq $ 2 , %rbp
 addq $ 8 , %rax
.endm

.macro SAVE8x2

 vbroadcastss 48(%rsp), %ymm0

 vmulps %ymm0 , %ymm4 , %ymm4
 vmulps %ymm0 , %ymm6 , %ymm6



 vaddps (%r15), %ymm4,%ymm4
 vaddps (%r15, %r10), %ymm6,%ymm6



 vmovups %ymm4 , (%r15)
 vmovups %ymm6 , (%r15, %r10)

.endm





.macro KERNEL4x2_SUB
 vmovups -16 * 4(%rdi, %rax, 4), %xmm0
 vbroadcastss -4 * 4(%rsi, %rbp, 4), %xmm2
 vbroadcastss -3 * 4(%rsi, %rbp, 4), %xmm3
 vfmadd231ps %xmm2,%xmm0,%xmm4
 vfmadd231ps %xmm3,%xmm0,%xmm6
 addq $ 2 , %rbp
 addq $ 4 , %rax
.endm

.macro SAVE4x2

 vbroadcastss 48(%rsp), %xmm0

 vmulps %xmm0 , %xmm4 , %xmm4
 vmulps %xmm0 , %xmm6 , %xmm6



 vaddps (%r15), %xmm4,%xmm4
 vaddps (%r15, %r10), %xmm6,%xmm6



 vmovups %xmm4 , (%r15)
 vmovups %xmm6 , (%r15, %r10)

.endm




.macro KERNEL2x2_SUB
 vmovss -16 * 4(%rdi, %rax, 4), %xmm0
 vmovss -15 * 4(%rdi, %rax, 4), %xmm1
 vmovss -4 * 4(%rsi, %rbp, 4), %xmm2
 vmovss -3 * 4(%rsi, %rbp, 4), %xmm3
 vfmadd231ss %xmm2,%xmm0,%xmm4
 vfmadd231ss %xmm2,%xmm1,%xmm5
 vfmadd231ss %xmm3,%xmm0,%xmm6
 vfmadd231ss %xmm3,%xmm1,%xmm7
 addq $ 2 , %rbp
 addq $ 2, %rax
.endm

.macro SAVE2x2

 vmovss 48(%rsp), %xmm0

 vmulss %xmm0 , %xmm4 , %xmm4
 vmulss %xmm0 , %xmm5 , %xmm5
 vmulss %xmm0 , %xmm6 , %xmm6
 vmulss %xmm0 , %xmm7 , %xmm7




 vaddss (%r15), %xmm4,%xmm4
 vaddss 1 * 4(%r15), %xmm5,%xmm5

 vaddss (%r15, %r10), %xmm6,%xmm6
 vaddss 1 * 4(%r15, %r10), %xmm7,%xmm7



 vmovss %xmm4 , (%r15)
 vmovss %xmm5 , 1 * 4(%r15)

 vmovss %xmm6 , (%r15, %r10)
 vmovss %xmm7 , 1 * 4(%r15, %r10)

.endm




.macro KERNEL1x2_SUB
 vmovss -16 * 4(%rdi, %rax, 4), %xmm0
 vmovss -4 * 4(%rsi, %rbp, 4), %xmm2
 vmovss -3 * 4(%rsi, %rbp, 4), %xmm3
 vfmadd231ss %xmm2,%xmm0,%xmm4
 vfmadd231ss %xmm3,%xmm0,%xmm6
 addq $ 2 , %rbp
 addq $ 1, %rax
.endm

.macro SAVE1x2

 vmovss 48(%rsp), %xmm0

 vmulss %xmm0 , %xmm4 , %xmm4
 vmulss %xmm0 , %xmm6 , %xmm6



 vaddss (%r15), %xmm4,%xmm4
 vaddss (%r15, %r10), %xmm6,%xmm6



 vmovss %xmm4 , (%r15)
 vmovss %xmm6 , (%r15, %r10)

.endm
# 987 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
.macro KERNEL16x1_SUB
 vmovups -16 * 4(%rdi, %rax, 4), %ymm0
 vmovups -8 * 4(%rdi, %rax, 4), %ymm1
 vbroadcastss -4 * 4(%rsi, %rbp, 4), %ymm2
 vfmadd231ps %ymm2,%ymm0,%ymm4
 vfmadd231ps %ymm2,%ymm1,%ymm5
 addq $ 1 , %rbp
 addq $ 16, %rax
.endm

.macro SAVE16x1

 vbroadcastss 48(%rsp), %ymm0

 vmulps %ymm0 , %ymm4 , %ymm4
 vmulps %ymm0 , %ymm5 , %ymm5



 vaddps (%r15), %ymm4,%ymm4
 vaddps 8 * 4(%r15), %ymm5,%ymm5



 vmovups %ymm4 , (%r15)
 vmovups %ymm5 , 8 * 4(%r15)

.endm




.macro KERNEL8x1_SUB
 vmovups -16 * 4(%rdi, %rax, 4), %ymm0
 vbroadcastss -4 * 4(%rsi, %rbp, 4), %ymm2
 vfmadd231ps %ymm2,%ymm0,%ymm4
 addq $ 1 , %rbp
 addq $ 8 , %rax
.endm

.macro SAVE8x1

 vbroadcastss 48(%rsp), %ymm0

 vmulps %ymm0 , %ymm4 , %ymm4



 vaddps (%r15), %ymm4,%ymm4



 vmovups %ymm4 , (%r15)

.endm





.macro KERNEL4x1_SUB
 vmovups -16 * 4(%rdi, %rax, 4), %xmm0
 vbroadcastss -4 * 4(%rsi, %rbp, 4), %xmm2
 vfmadd231ps %xmm2,%xmm0,%xmm4
 addq $ 1 , %rbp
 addq $ 4 , %rax
.endm

.macro SAVE4x1

 vbroadcastss 48(%rsp), %xmm0

 vmulps %xmm0 , %xmm4 , %xmm4



 vaddps (%r15), %xmm4,%xmm4



 vmovups %xmm4 , (%r15)

.endm




.macro KERNEL2x1_SUB
 vmovss -16 * 4(%rdi, %rax, 4), %xmm0
 vmovss -15 * 4(%rdi, %rax, 4), %xmm1
 vmovss -4 * 4(%rsi, %rbp, 4), %xmm2
 vfmadd231ss %xmm2,%xmm0,%xmm4
 vfmadd231ss %xmm2,%xmm1,%xmm5
 addq $ 1 , %rbp
 addq $ 2 , %rax
.endm

.macro SAVE2x1

 vmovss 48(%rsp), %xmm0

 vmulss %xmm0 , %xmm4 , %xmm4
 vmulss %xmm0 , %xmm5 , %xmm5



 vaddss (%r15), %xmm4,%xmm4
 vaddss 1 * 4(%r15), %xmm5,%xmm5



 vmovss %xmm4 , (%r15)
 vmovss %xmm5 , 1 * 4(%r15)

.endm




.macro KERNEL1x1_SUB
 vmovss -16 * 4(%rdi, %rax, 4), %xmm0
 vmovss -4 * 4(%rsi, %rbp, 4), %xmm2
 vfmadd231ss %xmm2,%xmm0,%xmm4
 addq $ 1 , %rbp
 addq $ 1 , %rax
.endm

.macro SAVE1x1

 vmovss 48(%rsp), %xmm0

 vmulss %xmm0 , %xmm4 , %xmm4



 vaddss (%r15), %xmm4,%xmm4



 vmovss %xmm4 , (%r15)

.endm
# 1140 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 .text; .align 512; .globl sgemm_kernel ; .type sgemm_kernel, @function; sgemm_kernel:


 subq $96, %rsp
 movq %rbx, (%rsp)
 movq %rbp, 8(%rsp)
 movq %r12, 16(%rsp)
 movq %r13, 24(%rsp)
 movq %r14, 32(%rsp)
 movq %r15, 40(%rsp)

 vzeroupper
# 1180 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 movq 96 + 8(%rsp), %r10






 movq %rsp, %rbx # save old stack
        subq $128 + 12288, %rsp
        andq $-4096, %rsp # align stack

       

 cmpq $0, %rdi
 je .L999

 cmpq $0, %rsi
 je .L999

 cmpq $0, %rdx
 je .L999

 movq %rdi, %r13
 movq %rsi, 40(%rsp)
 movq %rdx, %r12

 vmovss %xmm0, 48(%rsp)

 salq $2, %r10

 movq 40(%rsp), %rax
        xorq %rdx, %rdx
        movq $12, %rdi
        divq %rdi
        movq %rax, 24(%rsp)
        movq %rdx, 32(%rsp)

 movq 24(%rsp), %r14
 cmpq $0, %r14
 je .L4_00
 .align 16




.L6_01:

 movq %r8, %rdi
 leaq 128(%rsp), %rsi
 movq %r12, %rax
 salq $2, %rax
        leaq (%r8, %rax,4), %rbp
        movq %rbp, %r8
        movq %r12, %rax

 .align 16


.L6_02c:

 vmovups (%rdi), %xmm0
 vmovsd (%rbp), %xmm1
 vmovups %xmm0, (%rsi)
 vmovsd %xmm1, 4*4(%rsi)
 addq $ 4*4,%rdi
 addq $ 4*4,%rbp
 addq $ 6*4,%rsi
 decq %rax
 jnz .L6_02c


.L6_10:
 movq %r9, %r15
 leaq (%r9, %r10, 2), %rdx
 leaq (%rdx, %r10, 1), %rdx
 leaq (%r9, %r10, 4), %r9
 leaq (%r9, %r10, 2), %r9

 movq %rcx, %rdi
 addq $ 16 * 4, %rdi

 movq %r13, %r11
 sarq $4, %r11
 je .L6_20

 .align 16

.L6_11:
        leaq 128(%rsp), %rsi
        addq $4 * 4, %rsi

 vzeroall

        movq %r12, %rax

 andq $-8, %rax
 je .L6_16

 .align 16

.L6_12:

 KERNEL16x6_SUB
 KERNEL16x6_SUB
 KERNEL16x6_SUB
 KERNEL16x6_SUB

 KERNEL16x6_SUB
 KERNEL16x6_SUB
 KERNEL16x6_SUB
 KERNEL16x6_SUB

 je .L6_16

 KERNEL16x6_SUB
 KERNEL16x6_SUB
 KERNEL16x6_SUB
 KERNEL16x6_SUB

 KERNEL16x6_SUB
 KERNEL16x6_SUB
 KERNEL16x6_SUB
 KERNEL16x6_SUB

 je .L6_16

 jmp .L6_12
 .align 16

.L6_16:
        movq %r12, %rax

 andq $7, %rax # if (k & 1)
 je .L6_19

 .align 16

.L6_17:

 KERNEL16x6_SUB

 jnz .L6_17
 .align 16


.L6_19:

 SAVE16x6

 addq $16 * 4, %r15 # coffset += 16
 addq $16 * 4, %rdx # coffset += 16
 decq %r11 # i --
 jg .L6_11
 .align 16




.L6_20:


 testq $15, %r13
 jz .L6_60

 testq $8, %r13
 jz .L6_21pre
 .align 16



.L6_20_1:
        leaq 128(%rsp), %rsi
        addq $4 * 4, %rsi

 vzeroall

        movq %r12, %rax

 andq $-8, %rax
 je .L6_20_6

 .align 16

.L6_20_2:

 prefetcht0 512(%rdi)
 KERNEL8x6_SUB
 KERNEL8x6_SUB
 prefetcht0 512(%rdi)
 KERNEL8x6_SUB
 KERNEL8x6_SUB

 prefetcht0 512(%rdi)
 KERNEL8x6_SUB
 KERNEL8x6_SUB
 prefetcht0 512(%rdi)
 KERNEL8x6_SUB
 KERNEL8x6_SUB

 je .L6_20_6

 prefetcht0 512(%rdi)
 KERNEL8x6_SUB
 KERNEL8x6_SUB
 prefetcht0 512(%rdi)
 KERNEL8x6_SUB
 KERNEL8x6_SUB

 prefetcht0 512(%rdi)
 KERNEL8x6_SUB
 KERNEL8x6_SUB
 prefetcht0 512(%rdi)
 KERNEL8x6_SUB
 KERNEL8x6_SUB

 je .L6_20_6

 jmp .L6_20_2
 .align 16

.L6_20_6:
        movq %r12, %rax

 andq $7, %rax # if (k & 1)
 je .L6_20_9

 .align 16

.L6_20_7:

 KERNEL8x6_SUB

 jnz .L6_20_7
 .align 16


.L6_20_9:

 SAVE8x6

 addq $8 * 4, %r15 # coffset += 8
 addq $8 * 4, %rdx # coffset += 8
 .align 16





.L6_21pre:

 testq $4, %r13
 jz .L6_30
 .align 16

.L6_21:
        leaq 128(%rsp), %rsi
        addq $4 * 4, %rsi

 vzeroall

        movq %r12, %rax

 andq $-8, %rax
 je .L6_26

 .align 16

.L6_22:

 prefetcht0 512(%rdi)
 KERNEL4x6_SUB
 KERNEL4x6_SUB
 KERNEL4x6_SUB
 KERNEL4x6_SUB

 prefetcht0 512(%rdi)
 KERNEL4x6_SUB
 KERNEL4x6_SUB
 KERNEL4x6_SUB
 KERNEL4x6_SUB

 je .L6_26

 prefetcht0 512(%rdi)
 KERNEL4x6_SUB
 KERNEL4x6_SUB
 KERNEL4x6_SUB
 KERNEL4x6_SUB

 prefetcht0 512(%rdi)
 KERNEL4x6_SUB
 KERNEL4x6_SUB
 KERNEL4x6_SUB
 KERNEL4x6_SUB

 je .L6_26

 jmp .L6_22
 .align 16

.L6_26:
        movq %r12, %rax

 andq $7, %rax # if (k & 1)
 je .L6_29

 .align 16

.L6_27:

 KERNEL4x6_SUB

 jnz .L6_27
 .align 16


.L6_29:

 SAVE4x6

 addq $4 * 4, %r15 # coffset += 4
 addq $4 * 4, %rdx # coffset += 4
 .align 16


.L6_30:
 testq $2, %r13
 jz .L6_40

 .align 16

.L6_31:
        leaq 128(%rsp), %rsi
        addq $4 * 4, %rsi

 vzeroall

        movq %r12, %rax

 andq $-8, %rax
 je .L6_36

 .align 16

.L6_32:

 prefetcht0 512(%rdi)
 KERNEL2x6_SUB
 KERNEL2x6_SUB
 KERNEL2x6_SUB
 KERNEL2x6_SUB

 KERNEL2x6_SUB
 KERNEL2x6_SUB
 KERNEL2x6_SUB
 KERNEL2x6_SUB

 je .L6_36

 prefetcht0 512(%rdi)
 KERNEL2x6_SUB
 KERNEL2x6_SUB
 KERNEL2x6_SUB
 KERNEL2x6_SUB

 KERNEL2x6_SUB
 KERNEL2x6_SUB
 KERNEL2x6_SUB
 KERNEL2x6_SUB

 je .L6_36

 jmp .L6_32
 .align 16

.L6_36:
        movq %r12, %rax

 andq $7, %rax # if (k & 1)
 je .L6_39

 .align 16

.L6_37:

 KERNEL2x6_SUB

 jnz .L6_37
 .align 16


.L6_39:

 SAVE2x6

 addq $2 * 4, %r15 # coffset += 2
 addq $2 * 4, %rdx # coffset += 2
 .align 16

.L6_40:
 testq $1, %r13
 jz .L6_60

 .align 16

.L6_41:
        leaq 128(%rsp), %rsi
        addq $4 * 4, %rsi

 vzeroall

        movq %r12, %rax

 andq $-8, %rax
 je .L6_46

 .align 16

.L6_42:

 prefetcht0 512(%rdi)
 KERNEL1x6_SUB
 KERNEL1x6_SUB
 KERNEL1x6_SUB
 KERNEL1x6_SUB

 KERNEL1x6_SUB
 KERNEL1x6_SUB
 KERNEL1x6_SUB
 KERNEL1x6_SUB

 je .L6_46

 KERNEL1x6_SUB
 KERNEL1x6_SUB
 KERNEL1x6_SUB
 KERNEL1x6_SUB

 KERNEL1x6_SUB
 KERNEL1x6_SUB
 KERNEL1x6_SUB
 KERNEL1x6_SUB

 je .L6_46

 jmp .L6_42
 .align 16

.L6_46:
        movq %r12, %rax

 andq $7, %rax # if (k & 1)
 je .L6_49

 .align 16

.L6_47:

 KERNEL1x6_SUB

 jnz .L6_47
 .align 16


.L6_49:

 SAVE1x6

 addq $1 * 4, %r15 # coffset += 1
 addq $1 * 4, %rdx # coffset += 1
 .align 16





.L6_60:





.L7_01:

 movq %r8, %rdi
 leaq 128(%rsp), %rsi
 movq %r12, %rax
 salq $2, %rax
        leaq (%r8, %rax,4), %rbp
        movq %r12, %rax

 .align 16


.L7_02c:

 vmovsd 2*4(%rdi), %xmm0
 vmovups (%rbp), %xmm1
 vmovsd %xmm0, (%rsi)
 vmovups %xmm1, 2*4(%rsi)
 addq $ 4*4,%rdi
 addq $ 4*4,%rbp
 addq $ 6*4,%rsi
 decq %rax
 jnz .L7_02c

        movq %rbp, %r8

.L7_10:
 movq %r9, %r15
 leaq (%r9, %r10, 2), %rdx
 leaq (%rdx, %r10, 1), %rdx
 leaq (%r9, %r10, 4), %r9
 leaq (%r9, %r10, 2), %r9

 movq %rcx, %rdi
 addq $ 16 * 4, %rdi

 movq %r13, %r11
 sarq $4, %r11
 je .L7_20

 .align 16

.L7_11:
        leaq 128(%rsp), %rsi
        addq $4 * 4, %rsi

 vzeroall

        movq %r12, %rax

 andq $-8, %rax
 je .L7_16

 .align 16

.L7_12:

 KERNEL16x6_SUB
 KERNEL16x6_SUB
 KERNEL16x6_SUB
 KERNEL16x6_SUB

 KERNEL16x6_SUB
 KERNEL16x6_SUB
 KERNEL16x6_SUB
 KERNEL16x6_SUB

 je .L7_16

 KERNEL16x6_SUB
 KERNEL16x6_SUB
 KERNEL16x6_SUB
 KERNEL16x6_SUB

 KERNEL16x6_SUB
 KERNEL16x6_SUB
 KERNEL16x6_SUB
 KERNEL16x6_SUB

 je .L7_16

 jmp .L7_12
 .align 16

.L7_16:
        movq %r12, %rax

 andq $7, %rax # if (k & 1)
 je .L7_19

 .align 16

.L7_17:

 KERNEL16x6_SUB

 jnz .L7_17
 .align 16


.L7_19:

 SAVE16x6

 addq $16 * 4, %r15 # coffset += 16
 addq $16 * 4, %rdx # coffset += 16
 decq %r11 # i --
 jg .L7_11
 .align 16




.L7_20:


 testq $15, %r13
 jz .L7_60

 testq $8, %r13
 jz .L7_21pre
 .align 16



.L7_20_1:
        leaq 128(%rsp), %rsi
        addq $4 * 4, %rsi

 vzeroall

        movq %r12, %rax

 andq $-8, %rax
 je .L7_20_6

 .align 16

.L7_20_2:

 prefetcht0 512(%rdi)
 KERNEL8x6_SUB
 KERNEL8x6_SUB
 prefetcht0 512(%rdi)
 KERNEL8x6_SUB
 KERNEL8x6_SUB

 prefetcht0 512(%rdi)
 KERNEL8x6_SUB
 KERNEL8x6_SUB
 prefetcht0 512(%rdi)
 KERNEL8x6_SUB
 KERNEL8x6_SUB

 je .L7_20_6

 prefetcht0 512(%rdi)
 KERNEL8x6_SUB
 KERNEL8x6_SUB
 prefetcht0 512(%rdi)
 KERNEL8x6_SUB
 KERNEL8x6_SUB

 prefetcht0 512(%rdi)
 KERNEL8x6_SUB
 KERNEL8x6_SUB
 prefetcht0 512(%rdi)
 KERNEL8x6_SUB
 KERNEL8x6_SUB

 je .L7_20_6

 jmp .L7_20_2
 .align 16

.L7_20_6:
        movq %r12, %rax

 andq $7, %rax # if (k & 1)
 je .L7_20_9

 .align 16

.L7_20_7:

 KERNEL8x6_SUB

 jnz .L7_20_7
 .align 16


.L7_20_9:

 SAVE8x6

 addq $8 * 4, %r15 # coffset += 8
 addq $8 * 4, %rdx # coffset += 8
 .align 16





.L7_21pre:

 testq $4, %r13
 jz .L7_30
 .align 16

.L7_21:
        leaq 128(%rsp), %rsi
        addq $4 * 4, %rsi

 vzeroall

        movq %r12, %rax

 andq $-8, %rax
 je .L7_26

 .align 16

.L7_22:

 prefetcht0 512(%rdi)
 KERNEL4x6_SUB
 KERNEL4x6_SUB
 KERNEL4x6_SUB
 KERNEL4x6_SUB

 prefetcht0 512(%rdi)
 KERNEL4x6_SUB
 KERNEL4x6_SUB
 KERNEL4x6_SUB
 KERNEL4x6_SUB

 je .L7_26

 prefetcht0 512(%rdi)
 KERNEL4x6_SUB
 KERNEL4x6_SUB
 KERNEL4x6_SUB
 KERNEL4x6_SUB

 prefetcht0 512(%rdi)
 KERNEL4x6_SUB
 KERNEL4x6_SUB
 KERNEL4x6_SUB
 KERNEL4x6_SUB

 je .L7_26

 jmp .L7_22
 .align 16

.L7_26:
        movq %r12, %rax

 andq $7, %rax # if (k & 1)
 je .L7_29

 .align 16

.L7_27:

 KERNEL4x6_SUB

 jnz .L7_27
 .align 16


.L7_29:

 SAVE4x6

 addq $4 * 4, %r15 # coffset += 4
 addq $4 * 4, %rdx # coffset += 4
 .align 16


.L7_30:
 testq $2, %r13
 jz .L7_40

 .align 16

.L7_31:
        leaq 128(%rsp), %rsi
        addq $4 * 4, %rsi

 vzeroall

        movq %r12, %rax

 andq $-8, %rax
 je .L7_36

 .align 16

.L7_32:

 prefetcht0 512(%rdi)
 KERNEL2x6_SUB
 KERNEL2x6_SUB
 KERNEL2x6_SUB
 KERNEL2x6_SUB

 KERNEL2x6_SUB
 KERNEL2x6_SUB
 KERNEL2x6_SUB
 KERNEL2x6_SUB

 je .L7_36

 prefetcht0 512(%rdi)
 KERNEL2x6_SUB
 KERNEL2x6_SUB
 KERNEL2x6_SUB
 KERNEL2x6_SUB

 KERNEL2x6_SUB
 KERNEL2x6_SUB
 KERNEL2x6_SUB
 KERNEL2x6_SUB

 je .L7_36

 jmp .L7_32
 .align 16

.L7_36:
        movq %r12, %rax

 andq $7, %rax # if (k & 1)
 je .L7_39

 .align 16

.L7_37:

 KERNEL2x6_SUB

 jnz .L7_37
 .align 16


.L7_39:

 SAVE2x6

 addq $2 * 4, %r15 # coffset += 2
 addq $2 * 4, %rdx # coffset += 2
 .align 16

.L7_40:
 testq $1, %r13
 jz .L7_60

 .align 16

.L7_41:
        leaq 128(%rsp), %rsi
        addq $4 * 4, %rsi

 vzeroall

        movq %r12, %rax

 andq $-8, %rax
 je .L7_46

 .align 16

.L7_42:

 prefetcht0 512(%rdi)
 KERNEL1x6_SUB
 KERNEL1x6_SUB
 KERNEL1x6_SUB
 KERNEL1x6_SUB

 KERNEL1x6_SUB
 KERNEL1x6_SUB
 KERNEL1x6_SUB
 KERNEL1x6_SUB

 je .L7_46

 KERNEL1x6_SUB
 KERNEL1x6_SUB
 KERNEL1x6_SUB
 KERNEL1x6_SUB

 KERNEL1x6_SUB
 KERNEL1x6_SUB
 KERNEL1x6_SUB
 KERNEL1x6_SUB

 je .L7_46

 jmp .L7_42
 .align 16

.L7_46:
        movq %r12, %rax

 andq $7, %rax # if (k & 1)
 je .L7_49

 .align 16

.L7_47:

 KERNEL1x6_SUB

 jnz .L7_47
 .align 16


.L7_49:

 SAVE1x6

 addq $1 * 4, %r15 # coffset += 1
 addq $1 * 4, %rdx # coffset += 1
 .align 16





.L7_60:

 decq %r14
 jg .L6_01





.L4_00:

  movq 32(%rsp), %r14
        sarq $2, %r14
        cmpq $ 0, %r14
        je .L2_00
        .align 16


.L4_01:

 movq %r8, %rdi
 leaq 128(%rsp), %rsi
 movq %r12, %rax
 sarq $2, %rax
 jz .L4_01b
 .align 16


.L4_01a:
        prefetcht0 512(%rdi)
        prefetchw 512(%rsi)

 vmovups (%rdi), %xmm0
 vmovups 4*4(%rdi), %xmm1
 vmovups 8*4(%rdi), %xmm2
 vmovups 12*4(%rdi), %xmm3

 vmovups %xmm0, (%rsi)
 vmovups %xmm1, 4*4(%rsi)
 vmovups %xmm2, 8*4(%rsi)
 vmovups %xmm3,12*4(%rsi)

 addq $ 16*4,%rdi
 addq $ 16*4,%rsi
 decq %rax
 jnz .L4_01a


.L4_01b:

        movq %r12, %rax
        andq $3, %rax
        jz .L4_02d
        .align 16

.L4_02c:

 vmovups (%rdi), %xmm0
 vmovups %xmm0, (%rsi)
 addq $ 4*4,%rdi
 addq $ 4*4,%rsi
 decq %rax
 jnz .L4_02c

.L4_02d:

 movq %rdi, %r8

.L4_10:
 movq %r9, %r15
 leaq (%r9, %r10, 2), %rdx
 leaq (%r9, %r10, 4), %r9






 movq %rcx, %rdi
 addq $ 16 * 4, %rdi

 movq %r13, %r11
 sarq $4, %r11
 je .L4_20

 .align 16

.L4_11:



        leaq 128(%rsp), %rsi
        addq $4 * 4, %rsi
# 2198 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 vzeroall


        movq %r12, %rax
# 2216 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 andq $-8, %rax
 je .L4_16
 movq %rax, %rbp
        leaq (,%rbp,4) , %rbp

 salq $4, %rax
 leaq (%rdi, %rax, 4), %rdi
 leaq (%rsi, %rbp, 4), %rsi
 negq %rbp
 negq %rax
 .align 16

.L4_12:

 prefetcht0 512(%rdi, %rax, 4)
 prefetcht0 512(%rsi, %rbp , 4)
 KERNEL16x4_SUB
 prefetcht0 512(%rdi, %rax, 4)
 KERNEL16x4_SUB
 prefetcht0 512(%rdi, %rax, 4)
 KERNEL16x4_SUB
 prefetcht0 512(%rdi, %rax, 4)
 KERNEL16x4_SUB

 prefetcht0 512(%rdi, %rax, 4)
 prefetcht0 512(%rsi, %rbp , 4)
 KERNEL16x4_SUB
 prefetcht0 512(%rdi, %rax, 4)
 KERNEL16x4_SUB
 prefetcht0 512(%rdi, %rax, 4)
 KERNEL16x4_SUB
 prefetcht0 512(%rdi, %rax, 4)
 KERNEL16x4_SUB

 je .L4_16

 prefetcht0 512(%rdi, %rax, 4)
 prefetcht0 512(%rsi, %rbp , 4)
 KERNEL16x4_SUB
 prefetcht0 512(%rdi, %rax, 4)
 KERNEL16x4_SUB
 prefetcht0 512(%rdi, %rax, 4)
 KERNEL16x4_SUB
 prefetcht0 512(%rdi, %rax, 4)
 KERNEL16x4_SUB

 prefetcht0 512(%rdi, %rax, 4)
 prefetcht0 512(%rsi, %rbp , 4)
 KERNEL16x4_SUB
 prefetcht0 512(%rdi, %rax, 4)
 KERNEL16x4_SUB
 prefetcht0 512(%rdi, %rax, 4)
 KERNEL16x4_SUB
 prefetcht0 512(%rdi, %rax, 4)
 KERNEL16x4_SUB

 je .L4_16

 jmp .L4_12
 .align 16

.L4_16:

        movq %r12, %rax




 andq $7, %rax # if (k & 1)
 je .L4_19

 movq %rax, %rbp
        leaq (,%rbp,4), %rbp

 salq $4, %rax
 leaq (%rdi, %rax, 4), %rdi
 leaq (%rsi, %rbp, 4), %rsi
 negq %rbp
 negq %rax
 .align 16

.L4_17:

 KERNEL16x4_SUB

 jl .L4_17
 .align 16


.L4_19:

 SAVE16x4
# 2325 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 addq $16 * 4, %r15 # coffset += 16
 addq $16 * 4, %rdx # coffset += 16
 decq %r11 # i --
 jg .L4_11
 .align 16




.L4_20:


 testq $15, %r13
 jz .L4_60

 testq $8, %r13
 jz .L4_21pre
 .align 16



.L4_20_1:



        leaq 128(%rsp), %rsi
        addq $4 * 4, %rsi
# 2364 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 vzeroall


        movq %r12, %rax
# 2383 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 andq $-8, %rax
 je .L4_20_6
 movq %rax, %rbp
        leaq (,%rbp,4), %rbp

 salq $3, %rax
 leaq (%rdi, %rax, 4), %rdi
 leaq (%rsi, %rbp, 4), %rsi
 negq %rbp
 negq %rax
 .align 16

.L4_20_2:

 KERNEL8x4_SUB
 KERNEL8x4_SUB
 KERNEL8x4_SUB
 KERNEL8x4_SUB

 KERNEL8x4_SUB
 KERNEL8x4_SUB
 KERNEL8x4_SUB
 KERNEL8x4_SUB

 je .L4_20_6

 KERNEL8x4_SUB
 KERNEL8x4_SUB
 KERNEL8x4_SUB
 KERNEL8x4_SUB

 KERNEL8x4_SUB
 KERNEL8x4_SUB
 KERNEL8x4_SUB
 KERNEL8x4_SUB

 je .L4_20_6

 jmp .L4_20_2
 .align 16

.L4_20_6:

        movq %r12, %rax




 andq $7, %rax # if (k & 1)
 je .L4_20_9

 movq %rax, %rbp
        leaq (,%rbp,4), %rbp

 salq $3, %rax
 leaq (%rdi, %rax, 4), %rdi
 leaq (%rsi, %rbp, 4), %rsi
 negq %rbp
 negq %rax
 .align 16

.L4_20_7:

 KERNEL8x4_SUB

 jl .L4_20_7
 .align 16


.L4_20_9:

 SAVE8x4
# 2472 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 addq $8 * 4, %r15 # coffset += 8
 addq $8 * 4, %rdx # coffset += 8
 .align 16





.L4_21pre:

 testq $4, %r13
 jz .L4_30
 .align 16

.L4_21:



        leaq 128(%rsp), %rsi
        addq $4 * 4, %rsi
# 2504 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 vzeroall


        movq %r12, %rax
# 2523 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 andq $-8, %rax
 je .L4_26
 movq %rax, %rbp
        leaq (,%rbp,4), %rbp

 salq $2, %rax
 leaq (%rdi, %rax, 4), %rdi
 leaq (%rsi, %rbp, 4), %rsi
 negq %rbp
 negq %rax
 .align 16

.L4_22:

 KERNEL4x4_SUB
 KERNEL4x4_SUB
 KERNEL4x4_SUB
 KERNEL4x4_SUB

 KERNEL4x4_SUB
 KERNEL4x4_SUB
 KERNEL4x4_SUB
 KERNEL4x4_SUB

 je .L4_26

 KERNEL4x4_SUB
 KERNEL4x4_SUB
 KERNEL4x4_SUB
 KERNEL4x4_SUB

 KERNEL4x4_SUB
 KERNEL4x4_SUB
 KERNEL4x4_SUB
 KERNEL4x4_SUB

 je .L4_26

 jmp .L4_22
 .align 16

.L4_26:

        movq %r12, %rax




 andq $7, %rax # if (k & 1)
 je .L4_29

 movq %rax, %rbp
        leaq (,%rbp,4), %rbp

 salq $2, %rax
 leaq (%rdi, %rax, 4), %rdi
 leaq (%rsi, %rbp, 4), %rsi
 negq %rbp
 negq %rax
 .align 16

.L4_27:

 KERNEL4x4_SUB

 jl .L4_27
 .align 16


.L4_29:

 SAVE4x4
# 2612 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 addq $4 * 4, %r15 # coffset += 4
 addq $4 * 4, %rdx # coffset += 4
 .align 16


.L4_30:
 testq $2, %r13
 jz .L4_40

 .align 16

.L4_31:



        leaq 128(%rsp), %rsi
        addq $4 * 4, %rsi
# 2641 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 vzeroall


        movq %r12, %rax
# 2660 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 andq $-8, %rax
 je .L4_36
 movq %rax, %rbp
        leaq (,%rbp,4), %rbp

 salq $1, %rax
 leaq (%rdi, %rax, 4), %rdi
 leaq (%rsi, %rbp, 4), %rsi
 negq %rbp
 negq %rax
 .align 16

.L4_32:

 KERNEL2x4_SUB
 KERNEL2x4_SUB
 KERNEL2x4_SUB
 KERNEL2x4_SUB

 KERNEL2x4_SUB
 KERNEL2x4_SUB
 KERNEL2x4_SUB
 KERNEL2x4_SUB

 je .L4_36

 KERNEL2x4_SUB
 KERNEL2x4_SUB
 KERNEL2x4_SUB
 KERNEL2x4_SUB

 KERNEL2x4_SUB
 KERNEL2x4_SUB
 KERNEL2x4_SUB
 KERNEL2x4_SUB

 je .L4_36

 jmp .L4_32
 .align 16

.L4_36:

        movq %r12, %rax




 andq $7, %rax # if (k & 1)
 je .L4_39

 movq %rax, %rbp
 leaq (,%rbp, 4), %rbp

 salq $1, %rax
 leaq (%rdi, %rax, 4), %rdi
 leaq (%rsi, %rbp, 4), %rsi
 negq %rbp
 negq %rax
 .align 16

.L4_37:

 KERNEL2x4_SUB

 jl .L4_37
 .align 16


.L4_39:

 SAVE2x4
# 2749 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 addq $2 * 4, %r15 # coffset += 2
 addq $2 * 4, %rdx # coffset += 2
 .align 16

.L4_40:
 testq $1, %r13
 jz .L4_60

 .align 16

.L4_41:



        leaq 128(%rsp), %rsi
        addq $4 * 4, %rsi
# 2776 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 vzeroall


        movq %r12, %rax
# 2794 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 andq $-8, %rax
 je .L4_46
 movq %rax, %rbp
        leaq (,%rbp,4), %rbp

 leaq (%rdi, %rax, 4), %rdi
 leaq (%rsi, %rbp, 4), %rsi
 negq %rbp
 negq %rax
 .align 16

.L4_42:

 KERNEL1x4_SUB
 KERNEL1x4_SUB
 KERNEL1x4_SUB
 KERNEL1x4_SUB

 KERNEL1x4_SUB
 KERNEL1x4_SUB
 KERNEL1x4_SUB
 KERNEL1x4_SUB

 je .L4_46

 KERNEL1x4_SUB
 KERNEL1x4_SUB
 KERNEL1x4_SUB
 KERNEL1x4_SUB

 KERNEL1x4_SUB
 KERNEL1x4_SUB
 KERNEL1x4_SUB
 KERNEL1x4_SUB

 je .L4_46

 jmp .L4_42
 .align 16

.L4_46:

        movq %r12, %rax




 andq $7, %rax # if (k & 1)
 je .L4_49

 movq %rax, %rbp
        leaq (,%rbp,4), %rbp

 leaq (%rdi, %rax, 4), %rdi
 leaq (%rsi, %rbp, 4), %rsi
 negq %rbp
 negq %rax
 .align 16

.L4_47:

 KERNEL1x4_SUB

 jl .L4_47
 .align 16


.L4_49:

 SAVE1x4
# 2880 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 addq $1 * 4, %r15 # coffset += 1
 addq $1 * 4, %rdx # coffset += 1
 .align 16





.L4_60:




 decq %r14
 jg .L4_01




.L2_00:

 movq 32(%rsp), %r14
 andq $3, %r14
 je .L999

 movq 32(%rsp), %r14
 andq $2, %r14
 je .L1_0

.L2_01:


 movq %r8, %rdi
 leaq 128(%rsp), %rsi
 movq %r12, %rax
 sarq $2, %rax
 jz .L2_01b
 .align 16

.L2_01a:

 vmovsd (%rdi), %xmm0
 vmovsd 2*4(%rdi), %xmm1
 vmovsd 4*4(%rdi), %xmm2
 vmovsd 6*4(%rdi), %xmm3

 vmovsd %xmm0, (%rsi)
 vmovsd %xmm1, 2*4(%rsi)
 vmovsd %xmm2, 4*4(%rsi)
 vmovsd %xmm3, 6*4(%rsi)

 addq $8*4,%rdi
 addq $8*4,%rsi
 decq %rax
 jnz .L2_01a


.L2_01b:

        movq %r12, %rax
        andq $3, %rax
        jz .L2_02d
        .align 16

.L2_02c:

 vmovsd (%rdi), %xmm0
 vmovsd %xmm0, (%rsi)
 addq $2*4,%rdi
 addq $2*4,%rsi
 decq %rax
 jnz .L2_02c

.L2_02d:

 movq %rdi, %r8

.L2_10:
 movq %r9, %r15
 leaq (%r9, %r10, 2), %r9






 movq %rcx, %rdi
 addq $16 * 4, %rdi

 movq %r13, %r11
 sarq $4, %r11
 je .L2_20

 .align 16

.L2_11:



        leaq 128(%rsp), %rsi
        addq $4 * 4, %rsi
# 2993 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 vzeroall


        movq %r12, %rax
# 3011 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 andq $-8, %rax
 je .L2_16
 movq %rax, %rbp
        leaq (%rbp,%rbp,1), %rbp

 salq $4, %rax
 leaq (%rdi, %rax, 4), %rdi
 leaq (%rsi, %rbp, 4), %rsi
 negq %rbp
 negq %rax
 .align 16

.L2_12:

 KERNEL16x2_SUB
 KERNEL16x2_SUB
 KERNEL16x2_SUB
 KERNEL16x2_SUB

 KERNEL16x2_SUB
 KERNEL16x2_SUB
 KERNEL16x2_SUB
 KERNEL16x2_SUB

 je .L2_16

 KERNEL16x2_SUB
 KERNEL16x2_SUB
 KERNEL16x2_SUB
 KERNEL16x2_SUB

 KERNEL16x2_SUB
 KERNEL16x2_SUB
 KERNEL16x2_SUB
 KERNEL16x2_SUB

 je .L2_16

 jmp .L2_12
 .align 16

.L2_16:

        movq %r12, %rax




 andq $7, %rax # if (k & 1)
 je .L2_19

 movq %rax, %rbp
        leaq (%rbp,%rbp,1), %rbp

 salq $4, %rax
 leaq (%rdi, %rax, 4), %rdi
 leaq (%rsi, %rbp, 4), %rsi
 negq %rbp
 negq %rax
 .align 16

.L2_17:

 KERNEL16x2_SUB

 jl .L2_17
 .align 16


.L2_19:

 SAVE16x2
# 3100 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 addq $16 * 4, %r15 # coffset += 16
 decq %r11 # i --
 jg .L2_11
 .align 16




.L2_20:


 testq $15, %r13
 jz .L2_60

 testq $8, %r13
 jz .L2_21pre
 .align 16



.L2_20_1:



        leaq 128(%rsp), %rsi
        addq $4 * 4, %rsi
# 3138 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 vzeroall


        movq %r12, %rax
# 3157 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 andq $-8, %rax
 je .L2_20_6
 movq %rax, %rbp
        leaq (%rbp,%rbp,1), %rbp

 salq $3, %rax
 leaq (%rdi, %rax, 4), %rdi
 leaq (%rsi, %rbp, 4), %rsi
 negq %rbp
 negq %rax
 .align 16

.L2_20_2:


 KERNEL8x2_SUB
 KERNEL8x2_SUB
 KERNEL8x2_SUB
 KERNEL8x2_SUB

 KERNEL8x2_SUB
 KERNEL8x2_SUB
 KERNEL8x2_SUB
 KERNEL8x2_SUB

 je .L2_20_6

 KERNEL8x2_SUB
 KERNEL8x2_SUB
 KERNEL8x2_SUB
 KERNEL8x2_SUB

 KERNEL8x2_SUB
 KERNEL8x2_SUB
 KERNEL8x2_SUB
 KERNEL8x2_SUB

 je .L2_20_6

 jmp .L2_20_2
 .align 16

.L2_20_6:

        movq %r12, %rax




 andq $7, %rax # if (k & 1)
 je .L2_20_9

 movq %rax, %rbp
        leaq (%rbp,%rbp,1), %rbp

 salq $3, %rax
 leaq (%rdi, %rax, 4), %rdi
 leaq (%rsi, %rbp, 4), %rsi
 negq %rbp
 negq %rax
 .align 16

.L2_20_7:

 KERNEL8x2_SUB

 jl .L2_20_7
 .align 16


.L2_20_9:

 SAVE8x2
# 3247 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 addq $8 * 4, %r15 # coffset += 8
 .align 16





.L2_21pre:

 testq $4, %r13
 jz .L2_30
 .align 16

.L2_21:



        leaq 128(%rsp), %rsi
        addq $4 * 4, %rsi
# 3278 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 vzeroall


        movq %r12, %rax
# 3297 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 andq $-8, %rax
 je .L2_26
 movq %rax, %rbp
        leaq (%rbp,%rbp,1), %rbp

 salq $2, %rax
 leaq (%rdi, %rax, 4), %rdi
 leaq (%rsi, %rbp, 4), %rsi
 negq %rbp
 negq %rax
 .align 16

.L2_22:


 KERNEL4x2_SUB
 KERNEL4x2_SUB
 KERNEL4x2_SUB
 KERNEL4x2_SUB

 KERNEL4x2_SUB
 KERNEL4x2_SUB
 KERNEL4x2_SUB
 KERNEL4x2_SUB

 je .L2_26

 KERNEL4x2_SUB
 KERNEL4x2_SUB
 KERNEL4x2_SUB
 KERNEL4x2_SUB

 KERNEL4x2_SUB
 KERNEL4x2_SUB
 KERNEL4x2_SUB
 KERNEL4x2_SUB

 je .L2_26

 jmp .L2_22
 .align 16

.L2_26:

        movq %r12, %rax




 andq $7, %rax # if (k & 1)
 je .L2_29

 movq %rax, %rbp
        leaq (%rbp,%rbp,1), %rbp

 salq $2, %rax
 leaq (%rdi, %rax, 4), %rdi
 leaq (%rsi, %rbp, 4), %rsi
 negq %rbp
 negq %rax
 .align 16

.L2_27:

 KERNEL4x2_SUB

 jl .L2_27
 .align 16


.L2_29:

 SAVE4x2
# 3387 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 addq $4 * 4, %r15 # coffset += 4
 .align 16


.L2_30:
 testq $2, %r13
 jz .L2_40

 .align 16

.L2_31:



        leaq 128(%rsp), %rsi
        addq $4 * 4, %rsi
# 3415 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 vzeroall


        movq %r12, %rax
# 3434 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 andq $-8, %rax
 je .L2_36
 movq %rax, %rbp
        leaq (%rbp,%rbp,1), %rbp

 salq $1, %rax
 leaq (%rdi, %rax, 4), %rdi
 leaq (%rsi, %rbp, 4), %rsi
 negq %rbp
 negq %rax
 .align 16

.L2_32:

 KERNEL2x2_SUB
 KERNEL2x2_SUB
 KERNEL2x2_SUB
 KERNEL2x2_SUB

 KERNEL2x2_SUB
 KERNEL2x2_SUB
 KERNEL2x2_SUB
 KERNEL2x2_SUB

 je .L2_36

 KERNEL2x2_SUB
 KERNEL2x2_SUB
 KERNEL2x2_SUB
 KERNEL2x2_SUB

 KERNEL2x2_SUB
 KERNEL2x2_SUB
 KERNEL2x2_SUB
 KERNEL2x2_SUB

 je .L2_36

 jmp .L2_32
 .align 16

.L2_36:

        movq %r12, %rax




 andq $7, %rax # if (k & 1)
 je .L2_39

 movq %rax, %rbp
        leaq (%rbp,%rbp,1), %rbp

 salq $1, %rax
 leaq (%rdi, %rax, 4), %rdi
 leaq (%rsi, %rbp, 4), %rsi
 negq %rbp
 negq %rax
 .align 16

.L2_37:

 KERNEL2x2_SUB

 jl .L2_37
 .align 16


.L2_39:

 SAVE2x2
# 3523 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 addq $2 * 4, %r15 # coffset += 2
 .align 16

.L2_40:
 testq $1, %r13
 jz .L2_60

 .align 16

.L2_41:



        leaq 128(%rsp), %rsi
        addq $4 * 4, %rsi
# 3549 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 vzeroall


        movq %r12, %rax
# 3567 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 andq $-8, %rax
 je .L2_46
 movq %rax, %rbp
        leaq (%rbp,%rbp,1), %rbp

 leaq (%rdi, %rax, 4), %rdi
 leaq (%rsi, %rbp, 4), %rsi
 negq %rbp
 negq %rax
 .align 16

.L2_42:

 KERNEL1x2_SUB
 KERNEL1x2_SUB
 KERNEL1x2_SUB
 KERNEL1x2_SUB

 KERNEL1x2_SUB
 KERNEL1x2_SUB
 KERNEL1x2_SUB
 KERNEL1x2_SUB

 je .L2_46

 KERNEL1x2_SUB
 KERNEL1x2_SUB
 KERNEL1x2_SUB
 KERNEL1x2_SUB

 KERNEL1x2_SUB
 KERNEL1x2_SUB
 KERNEL1x2_SUB
 KERNEL1x2_SUB

 je .L2_46

 jmp .L2_42
 .align 16

.L2_46:

        movq %r12, %rax




 andq $7, %rax # if (k & 1)
 je .L2_49

 movq %rax, %rbp
        leaq (%rbp,%rbp,1), %rbp

 leaq (%rdi, %rax, 4), %rdi
 leaq (%rsi, %rbp, 4), %rsi
 negq %rbp
 negq %rax
 .align 16

.L2_47:

 KERNEL1x2_SUB

 jl .L2_47
 .align 16


.L2_49:

 SAVE1x2
# 3653 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 addq $1 * 4, %r15 # coffset += 1
 .align 16





.L2_60:







.L1_0:





 movq 32(%rsp), %r14
 andq $1, %r14
 je .L999
 .align 16

.L1_01:

 movq %r8, %rdi
 leaq 128(%rsp), %rsi
 movq %r12, %rax
 .align 16

.L1_02b:

 vmovss (%rdi), %xmm0
 vmovss %xmm0, (%rsi)
 addq $1*4,%rdi
 addq $1*4,%rsi
 decq %rax
 jnz .L1_02b

.L1_02c:

 movq %rdi, %r8

.L1_10:
 movq %r9, %r15
 leaq (%r9, %r10, 1), %r9






 movq %rcx, %rdi
 addq $16 * 4, %rdi

 movq %r13, %r11
 sarq $4, %r11
 je .L1_20

 .align 16

.L1_11:



        leaq 128(%rsp), %rsi
        addq $4 * 4, %rsi
# 3734 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 vzeroall


        movq %r12, %rax
# 3752 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 andq $-8, %rax
 je .L1_16
 movq %rax, %rbp

 salq $4, %rax
 leaq (%rdi, %rax, 4), %rdi
 leaq (%rsi, %rbp, 4), %rsi
 negq %rbp
 negq %rax
 .align 16

.L1_12:

 KERNEL16x1_SUB
 KERNEL16x1_SUB
 KERNEL16x1_SUB
 KERNEL16x1_SUB

 KERNEL16x1_SUB
 KERNEL16x1_SUB
 KERNEL16x1_SUB
 KERNEL16x1_SUB

 je .L1_16

 KERNEL16x1_SUB
 KERNEL16x1_SUB
 KERNEL16x1_SUB
 KERNEL16x1_SUB

 KERNEL16x1_SUB
 KERNEL16x1_SUB
 KERNEL16x1_SUB
 KERNEL16x1_SUB

 je .L1_16

 jmp .L1_12
 .align 16

.L1_16:

        movq %r12, %rax




 andq $7, %rax # if (k & 1)
 je .L1_19

 movq %rax, %rbp

 salq $4, %rax
 leaq (%rdi, %rax, 4), %rdi
 leaq (%rsi, %rbp, 4), %rsi
 negq %rbp
 negq %rax
 .align 16

.L1_17:

 KERNEL16x1_SUB

 jl .L1_17
 .align 16


.L1_19:

 SAVE16x1
# 3838 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 addq $16 * 4, %r15 # coffset += 16
 decq %r11 # i --
 jg .L1_11
 .align 16




.L1_20:


 testq $15, %r13
 jz .L999

 testq $8, %r13
 jz .L1_21pre
 .align 16



.L1_20_1:



        leaq 128(%rsp), %rsi
        addq $4 * 4, %rsi
# 3875 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 vzeroall


        movq %r12, %rax
# 3894 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 andq $-8, %rax
 je .L1_20_6
 movq %rax, %rbp

 salq $3, %rax
 leaq (%rdi, %rax, 4), %rdi
 leaq (%rsi, %rbp, 4), %rsi
 negq %rbp
 negq %rax
 .align 16

.L1_20_2:

 KERNEL8x1_SUB
 KERNEL8x1_SUB
 KERNEL8x1_SUB
 KERNEL8x1_SUB

 KERNEL8x1_SUB
 KERNEL8x1_SUB
 KERNEL8x1_SUB
 KERNEL8x1_SUB

 je .L1_20_6

 KERNEL8x1_SUB
 KERNEL8x1_SUB
 KERNEL8x1_SUB
 KERNEL8x1_SUB

 KERNEL8x1_SUB
 KERNEL8x1_SUB
 KERNEL8x1_SUB
 KERNEL8x1_SUB

 je .L1_20_6

 jmp .L1_20_2
 .align 16

.L1_20_6:

        movq %r12, %rax




 andq $7, %rax # if (k & 1)
 je .L1_20_9

 movq %rax, %rbp

 salq $3, %rax
 leaq (%rdi, %rax, 4), %rdi
 leaq (%rsi, %rbp, 4), %rsi
 negq %rbp
 negq %rax
 .align 16

.L1_20_7:

 KERNEL8x1_SUB

 jl .L1_20_7
 .align 16


.L1_20_9:

 SAVE8x1
# 3980 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 addq $8 * 4, %r15 # coffset += 8
 .align 16





.L1_21pre:

 testq $4, %r13
 jz .L1_30
 .align 16

.L1_21:



        leaq 128(%rsp), %rsi
        addq $4 * 4, %rsi
# 4010 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 vzeroall


        movq %r12, %rax
# 4029 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 andq $-8, %rax
 je .L1_26
 movq %rax, %rbp

 salq $2, %rax
 leaq (%rdi, %rax, 4), %rdi
 leaq (%rsi, %rbp, 4), %rsi
 negq %rbp
 negq %rax
 .align 16

.L1_22:

 KERNEL4x1_SUB
 KERNEL4x1_SUB
 KERNEL4x1_SUB
 KERNEL4x1_SUB

 KERNEL4x1_SUB
 KERNEL4x1_SUB
 KERNEL4x1_SUB
 KERNEL4x1_SUB

 je .L1_26

 KERNEL4x1_SUB
 KERNEL4x1_SUB
 KERNEL4x1_SUB
 KERNEL4x1_SUB

 KERNEL4x1_SUB
 KERNEL4x1_SUB
 KERNEL4x1_SUB
 KERNEL4x1_SUB

 je .L1_26

 jmp .L1_22
 .align 16

.L1_26:

        movq %r12, %rax




 andq $7, %rax # if (k & 1)
 je .L1_29

 movq %rax, %rbp

 salq $2, %rax
 leaq (%rdi, %rax, 4), %rdi
 leaq (%rsi, %rbp, 4), %rsi
 negq %rbp
 negq %rax
 .align 16

.L1_27:

 KERNEL4x1_SUB

 jl .L1_27
 .align 16


.L1_29:

 SAVE4x1
# 4115 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 addq $4 * 4, %r15 # coffset += 4
 .align 16


.L1_30:
 testq $2, %r13
 jz .L1_40

 .align 16

.L1_31:



        leaq 128(%rsp), %rsi
        addq $4 * 4, %rsi
# 4142 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 vzeroall


        movq %r12, %rax
# 4161 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 andq $-8, %rax
 je .L1_36
 movq %rax, %rbp

 salq $1, %rax
 leaq (%rdi, %rax, 4), %rdi
 leaq (%rsi, %rbp, 4), %rsi
 negq %rbp
 negq %rax
 .align 16

.L1_32:

 KERNEL2x1_SUB
 KERNEL2x1_SUB
 KERNEL2x1_SUB
 KERNEL2x1_SUB

 KERNEL2x1_SUB
 KERNEL2x1_SUB
 KERNEL2x1_SUB
 KERNEL2x1_SUB

 je .L1_36

 KERNEL2x1_SUB
 KERNEL2x1_SUB
 KERNEL2x1_SUB
 KERNEL2x1_SUB

 KERNEL2x1_SUB
 KERNEL2x1_SUB
 KERNEL2x1_SUB
 KERNEL2x1_SUB

 je .L1_36

 jmp .L1_32
 .align 16

.L1_36:

        movq %r12, %rax




 andq $7, %rax # if (k & 1)
 je .L1_39

 movq %rax, %rbp

 salq $1, %rax
 leaq (%rdi, %rax, 4), %rdi
 leaq (%rsi, %rbp, 4), %rsi
 negq %rbp
 negq %rax
 .align 16

.L1_37:

 KERNEL2x1_SUB

 jl .L1_37
 .align 16


.L1_39:

 SAVE2x1
# 4247 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 addq $2 * 4, %r15 # coffset += 2
 .align 16

.L1_40:
 testq $1, %r13
 jz .L999

 .align 16

.L1_41:



        leaq 128(%rsp), %rsi
        addq $4 * 4, %rsi
# 4272 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 vzeroall


        movq %r12, %rax
# 4290 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 andq $-8, %rax
 je .L1_46
 movq %rax, %rbp

 leaq (%rdi, %rax, 4), %rdi
 leaq (%rsi, %rbp, 4), %rsi
 negq %rbp
 negq %rax
 .align 16

.L1_42:

 KERNEL1x1_SUB
 KERNEL1x1_SUB
 KERNEL1x1_SUB
 KERNEL1x1_SUB

 KERNEL1x1_SUB
 KERNEL1x1_SUB
 KERNEL1x1_SUB
 KERNEL1x1_SUB

 je .L1_46

 KERNEL1x1_SUB
 KERNEL1x1_SUB
 KERNEL1x1_SUB
 KERNEL1x1_SUB

 KERNEL1x1_SUB
 KERNEL1x1_SUB
 KERNEL1x1_SUB
 KERNEL1x1_SUB

 je .L1_46

 jmp .L1_42
 .align 16

.L1_46:

        movq %r12, %rax




 andq $7, %rax # if (k & 1)
 je .L1_49

 movq %rax, %rbp

 leaq (%rdi, %rax, 4), %rdi
 leaq (%rsi, %rbp, 4), %rsi
 negq %rbp
 negq %rax
 .align 16

.L1_47:

 KERNEL1x1_SUB

 jl .L1_47
 .align 16


.L1_49:

 SAVE1x1
# 4373 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 addq $1 * 4, %r15 # coffset += 1
 .align 16


.L999:
 movq %rbx, %rsp
 movq (%rsp), %rbx
 movq 8(%rsp), %rbp
 movq 16(%rsp), %r12
 movq 24(%rsp), %r13
 movq 32(%rsp), %r14
 movq 40(%rsp), %r15
# 4401 "../kernel/x86_64/sgemm_kernel_16x4_haswell.S"
 addq $96, %rsp
 ret

 .size sgemm_kernel, .-sgemm_kernel; .section .note.GNU-stack,"",@progbits
