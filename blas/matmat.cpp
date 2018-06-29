/*********************************************************************/
/* Copyright 2018 Andrea Nobile                                      */
/* All rights reserved.                                              */
/*                                                                   */
/* Redistribution and use in source and binary forms, with or        */
/* without modification, are permitted provided that the following   */
/* conditions are met:                                               */
/*                                                                   */
/*   1. Redistributions of source code must retain the above         */
/*      copyright notice, this list of conditions and the following  */
/*      disclaimer.                                                  */
/*                                                                   */
/*   2. Redistributions in binary form must reproduce the above      */
/*      copyright notice, this list of conditions and the following  */
/*      disclaimer in the documentation and/or other materials       */
/*      provided with the distribution.                              */
/*                                                                   */
/*    THIS  SOFTWARE IS PROVIDED  BY THE  UNIVERSITY OF  TEXAS AT    */
/*    AUSTIN  ``AS IS''  AND ANY  EXPRESS OR  IMPLIED WARRANTIES,    */
/*    INCLUDING, BUT  NOT LIMITED  TO, THE IMPLIED  WARRANTIES OF    */
/*    MERCHANTABILITY  AND FITNESS FOR  A PARTICULAR  PURPOSE ARE    */
/*    DISCLAIMED.  IN  NO EVENT SHALL THE UNIVERSITY  OF TEXAS AT    */
/*    AUSTIN OR CONTRIBUTORS BE  LIABLE FOR ANY DIRECT, INDIRECT,    */
/*    INCIDENTAL,  SPECIAL, EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES    */
/*    (INCLUDING, BUT  NOT LIMITED TO,  PROCUREMENT OF SUBSTITUTE    */
/*    GOODS  OR  SERVICES; LOSS  OF  USE,  DATA,  OR PROFITS;  OR    */
/*    BUSINESS INTERRUPTION) HOWEVER CAUSED  AND ON ANY THEORY OF    */
/*    LIABILITY, WHETHER  IN CONTRACT, STRICT  LIABILITY, OR TORT    */
/*    (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY WAY OUT    */
/*    OF  THE  USE OF  THIS  SOFTWARE,  EVEN  IF ADVISED  OF  THE    */
/*    POSSIBILITY OF SUCH DAMAGE.                                    */
/*                                                                   */
/*********************************************************************/

/*********************************************************************/
/* Copyright 2009, 2010 The University of Texas at Austin.           */
/* All rights reserved.                                              */
/*                                                                   */
/* Redistribution and use in source and binary forms, with or        */
/* without modification, are permitted provided that the following   */
/* conditions are met:                                               */
/*                                                                   */
/*   1. Redistributions of source code must retain the above         */
/*      copyright notice, this list of conditions and the following  */
/*      disclaimer.                                                  */
/*                                                                   */
/*   2. Redistributions in binary form must reproduce the above      */
/*      copyright notice, this list of conditions and the following  */
/*      disclaimer in the documentation and/or other materials       */
/*      provided with the distribution.                              */
/*                                                                   */
/*    THIS  SOFTWARE IS PROVIDED  BY THE  UNIVERSITY OF  TEXAS AT    */
/*    AUSTIN  ``AS IS''  AND ANY  EXPRESS OR  IMPLIED WARRANTIES,    */
/*    INCLUDING, BUT  NOT LIMITED  TO, THE IMPLIED  WARRANTIES OF    */
/*    MERCHANTABILITY  AND FITNESS FOR  A PARTICULAR  PURPOSE ARE    */
/*    DISCLAIMED.  IN  NO EVENT SHALL THE UNIVERSITY  OF TEXAS AT    */
/*    AUSTIN OR CONTRIBUTORS BE  LIABLE FOR ANY DIRECT, INDIRECT,    */
/*    INCIDENTAL,  SPECIAL, EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES    */
/*    (INCLUDING, BUT  NOT LIMITED TO,  PROCUREMENT OF SUBSTITUTE    */
/*    GOODS  OR  SERVICES; LOSS  OF  USE,  DATA,  OR PROFITS;  OR    */
/*    BUSINESS INTERRUPTION) HOWEVER CAUSED  AND ON ANY THEORY OF    */
/*    LIABILITY, WHETHER  IN CONTRACT, STRICT  LIABILITY, OR TORT    */
/*    (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY WAY OUT    */
/*    OF  THE  USE OF  THIS  SOFTWARE,  EVEN  IF ADVISED  OF  THE    */
/*    POSSIBILITY OF SUCH DAMAGE.                                    */
/*                                                                   */
/* The views and conclusions contained in the software and           */
/* documentation are those of the authors and should not be          */
/* interpreted as representing official policies, either expressed   */
/* or implied, of The University of Texas at Austin.                 */
/*********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <set>
#include "matmat.h"
#include "memory.h"


using namespace std;

extern "C"
{
int sgemm_itcopy(long m, long n, float *a, long lda, float *b);
int sgemm_kernel(long m, long n, long k, float alpha, float *a, float *b, float *c, long ldc);
int sgemm_oncopy(long m, long n, float *a, long lda, float *b);
}


#define GEMM_P 768
#define GEMM_Q 384
#define GEMM_R 21056

#define GEMM_UNROLL_M 16
#define GEMM_UNROLL_N 4


struct blas_arg_t {
    float *a, *b, *c;
    long m, n, k, lda, ldb, ldc;
};


static int sgemm_nn(blas_arg_t *args, float *sa, float *sb, long mode, float *bb)
{
    long k, lda, ldb, ldc;
    float alpha = 1.0f;
    float *a, *b, *c;
    long m, n;
    // m = num col a, num col c
    // n = num rows b, num_rows c

    long sra_scb, is, srbo_srco;
    //rbo_rco = row b outer, row c outer
    long rba_cbb, cba, rc_todo;

    long srb_src, rbb;

    long l1stride, gemm_p, l2size;

    a = args->a;
    b = args->b;
    c = args->c;

    lda = args->lda;
    ldb = args->ldb;
    ldc = args->ldc;

    m = args->m;
    n = args->n;
    k = args->k;

    int tot_b_sz = 0;

    if (k == 0) return 0;

    l2size = GEMM_P * GEMM_Q;
    for(srbo_srco = 0; srbo_srco < n; srbo_srco += GEMM_R){
        assert(srbo_srco == 0);
        rc_todo = n - srbo_srco;
        if (rc_todo > GEMM_R) rc_todo = GEMM_R;

        for(sra_scb = 0; sra_scb < k; sra_scb += rba_cbb){

            rba_cbb = k - sra_scb;

            if (rba_cbb >= GEMM_Q * 2) {
                gemm_p = GEMM_P;
                rba_cbb  = GEMM_Q;
            } else {
                if (rba_cbb > GEMM_Q) {
                    rba_cbb = ((rba_cbb / 2 + GEMM_UNROLL_M - 1)/GEMM_UNROLL_M) * GEMM_UNROLL_M;
                }
                gemm_p = ((l2size / rba_cbb + GEMM_UNROLL_M - 1)/GEMM_UNROLL_M) * GEMM_UNROLL_M;
                while (gemm_p * rba_cbb > l2size) gemm_p -= GEMM_UNROLL_M;
            }


            cba = m;
            l1stride = 1;


            if (cba >= gemm_p * 2) {
                cba = GEMM_P;
            } else {
                if (cba > GEMM_P) {
                    cba = ((cba / 2 + GEMM_UNROLL_M - 1)/GEMM_UNROLL_M) * GEMM_UNROLL_M;
                } else {
                    l1stride = 0;
                }
            }

            long bboffs = tot_b_sz;

            sgemm_itcopy(rba_cbb, cba, (float *)(a) + ((sra_scb) * (lda)), lda, sa);

            for(srb_src = srbo_srco; srb_src < srbo_srco + rc_todo; srb_src += rbb){
                rbb = rc_todo + srbo_srco - srb_src;

                if (rbb >= 3*GEMM_UNROLL_N) rbb = 3*GEMM_UNROLL_N;
                else
                    if (rbb >= 2*GEMM_UNROLL_N) rbb = 2*GEMM_UNROLL_N;
                    else
                        if (rbb > GEMM_UNROLL_N) rbb = GEMM_UNROLL_N;

                if(mode == 0) {

                    sgemm_oncopy(rba_cbb, rbb, (float *)(b) + ((sra_scb) + srb_src*ldb),
                                 ldb, sb + rba_cbb * (srb_src - srbo_srco) * l1stride);
                    sgemm_kernel(cba, rbb, rba_cbb, alpha, sa, sb + rba_cbb * (srb_src - srbo_srco) * l1stride,
                                 c + srb_src*ldc, ldc);
                    tot_b_sz += rba_cbb*rbb;

                } else if(mode == 1) {

                    sgemm_oncopy(rba_cbb, rbb, (float *)(b) + ((sra_scb) + srb_src*ldb),
                                 ldb, bb + tot_b_sz);
                    sgemm_kernel(cba, rbb, rba_cbb, alpha, sa, bb + tot_b_sz,
                                 c + srb_src*ldc, ldc);
                    tot_b_sz += rba_cbb*rbb;

                } else if(mode == 2) {

                    sgemm_kernel(cba, rbb, rba_cbb, alpha, sa, bb + tot_b_sz,
                                 c + srb_src*ldc, ldc);
                    tot_b_sz += rba_cbb*rbb;

                }

            }

            for(is = cba; is < m; is += cba){
                cba = m - is;

                if (cba >= GEMM_P * 2) {
                    cba = GEMM_P;
                } else {
                    if (cba > GEMM_P) {
                        cba = ((cba / 2 + GEMM_UNROLL_M - 1)/GEMM_UNROLL_M) * GEMM_UNROLL_M;
                    }
                }

                sgemm_itcopy(rba_cbb, cba, a + (is + sra_scb*lda), lda, sa);;

                if(mode == 0) {
                    sgemm_kernel(cba, rc_todo, rba_cbb, alpha, sa, sb, c + (is + srbo_srco*ldc), ldc);
                } else {
                    sgemm_kernel(cba, rc_todo, rba_cbb, alpha, sa, bb+bboffs, c + (is + srbo_srco*ldc), ldc);
                }

            }
        }
    }
    assert(tot_b_sz == n*k);
    return 0;
}


static set<float*> set_a;

void matmat(float *a, float *b, float *c, int m, int n, int k, int mode, volatile unsigned long *lock_addr)
{
    blas_arg_t args;

    args.m = n;
    args.n = m;
    args.k = k;

    args.a = b;
    args.b = a;

    args.c = c;

    args.lda = n;
    args.ldb = k;
    args.ldc = n;

    assert(args.ldc >= args.m);
    assert(args.ldb >= args.k);
    assert(args.lda >= args.m);
    assert(args.k >= 0);
    assert(args.n >= 0);
    assert(args.m >= 0);

    if ((args.m == 0) || (args.n == 0)) return;

    float *buffer = (float *)blas_memory_alloc(0);

    float *sa = (float *)((long)buffer +0);
    float *sb = (float *)(((long)sa + ((GEMM_P * GEMM_Q * 1 * 4 + 0x03fffUL) & ~0x03fffUL)) + 0);

    memset(c, 0, (size_t)m*(size_t)n*sizeof(float));

    if(mode != 0) {  
        auto it = set_a.find(a);
        if(it != set_a.end()) {

            sgemm_nn(&args, sa, sb, 2, a);

        } else {
            if(lock_addr) {
                blas_lock(lock_addr);
            }

            auto it = set_a.find(a);
            if(it != set_a.end()) {

                sgemm_nn(&args, sa, sb, 2, a);
                blas_memory_free(buffer);
                return;
            }

            float *reordered_a = (float*) malloc(sizeof(float)*m*k);

            sgemm_nn(&args, sa, sb, 1, reordered_a);

            memcpy(a, reordered_a, sizeof(float)*m*k);
            set_a.insert(a);

            if(lock_addr) {
                blas_unlock(lock_addr);
            }
            free(reordered_a);
        }

    } else {
        sgemm_nn(&args, sa, sb, 0, NULL);
    }

    blas_memory_free(buffer);
}
