/********************************************************************************
 *  Copyright (C) <2018>  <Andrea Nobile>                                       *
 *                                                                              *
 *   This program is free software: you can redistribute it and/or modify       *
 *   it under the terms of the GNU Affero General Public License as             *
 *   published by the Free Software Foundation, either version 3 of the         *
 *   License, or (at your option) any later version.                            *
 *                                                                              *
 *  This program is distributed in the hope that it will be useful,             *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 *   GNU Affero General Public License for more details.                        *
 *                                                                              *
 *  You should have received a copy of the GNU Affero General Public License    *
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.      *
 *                                                                              *
 *******************************************************************************/

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include <immintrin.h>
#include <stdio.h>
#include "kernels.h"


#define v8sf __m256

#define avx_addps(a, b) _mm256_add_ps((a), (b))
#define avx_mulps(a, b) _mm256_mul_ps((a), (b))
#define avx_madd(a, b, c) _mm256_fmadd_ps((a), (b), (c))


float dot_reference(const float *restrict v0, const float *restrict v1, int64_t n)
{
    int64_t i;
    float tmp = 0.0f;
    for(i=0;i<n;i++) {
        tmp += v0[i]*v1[i];
    }
    return tmp;
}


void matvec(const float *restrict m, const float *restrict inv, float *restrict outv, int32_t matrix_nrows, int32_t matrix_ncols)
{
    int32_t i;

    for(i=0;i<matrix_nrows;i++) {
        const float *restrict prow = &m[matrix_ncols*i];
        outv[i] = dot(prow, inv, matrix_ncols);
    }
}


void test_dot()
{
    int len = 109;
    float *v0 = malloc(len*sizeof(float));
    float *v1 = malloc(len*sizeof(float));
    //assert((int64_t)v0%32 == 0);
    //assert((int64_t)v0%32 == 0);

    int i;
    for(i=0;i<len;i++) {
        v0[i] = 1.0;//drand48();
        v1[i] = 1.0;//drand48();
    }

    float ref = dot_reference(v0, v1, len);
    printf("ref = %f \n", ref);
    float act = dot(v0, v1, len);
    printf("act = %f \n", act);

    free(v0);
    free(v1);
}
