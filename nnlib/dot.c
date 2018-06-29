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

#include <immintrin.h>
#include <stdint.h>

#define v8sf __m256

#define avx_addps(a, b) _mm256_add_ps((a), (b))
#define avx_mulps(a, b) _mm256_mul_ps((a), (b))
#define avx_madd(a, b, c) _mm256_fmadd_ps((a), (b), (c))


float dot(const float *restrict v0, const float *restrict v1, size_t n)
{
    size_t i;

    if(n < 16) {
        float tmp = 0.0f;
        for(i=0;i<n;i++) {
            tmp += v0[i]*v1[i];
        }
        return tmp;
    }

    v8sf vout0, vout1;
    vout0 = _mm256_setzero_ps();
    vout1 = _mm256_setzero_ps();

    uint64_t n16 = n >> 4;

    v8sf v0_0, v0_1, v1_0, v1_1;

    for(i=n16;i>0;i--) {
        v0_0 = _mm256_loadu_ps(v0 + 0);
        v0_1 = _mm256_loadu_ps(v0 + 8);

        v1_0 = _mm256_loadu_ps(v1 + 0);
        v1_1 = _mm256_loadu_ps(v1 + 8);

        vout0 = avx_madd(v0_0, v1_0, vout0);
        vout1 = avx_madd(v0_1, v1_1, vout1);
        v0+=16;
        v1+=16;
    }


    vout0 = avx_addps(vout0, vout1);


    float *pvout = (float*) &vout0;
    float vsum = pvout[0] + pvout[1] + pvout[2] + pvout[3] +
            pvout[4] + pvout[5] + pvout[6] + pvout[7];


    float tmp = 0.0f;
    for(i=0;i<(n&15);i++) {
        tmp += v0[i]*v1[i];
    }
    return tmp + vsum;
}
