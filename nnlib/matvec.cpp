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
#include <vector>

using namespace std;

#include <immintrin.h>
#include <stdio.h>
#include "kernels.h"


float dot_reference(const float * __restrict__ v0, const float * __restrict__ v1, size_t n)
{
    size_t i;
    float tmp = 0.0f;
    for(i=0;i<n;i++) {
        tmp += v0[i]*v1[i];
    }
    return tmp;
}


void matvec(const float * __restrict__ m, const float * __restrict__ inv, float * __restrict__ outv, int32_t matrix_nrows, int32_t matrix_ncols)
{
    int i;

    for(i=0;i<matrix_nrows;i++) {
        const float * __restrict__ prow = &m[matrix_ncols*i];
        outv[i] = dot(prow, inv, matrix_ncols);
    }
}


void test_dot()
{
    int len = 109;

    vector<float> v0(len);
    vector<float> v1(len);

    int i;
    for(i=0;i<len;i++) {
        v0[i] = drand48();
        v1[i] = drand48();
    }

    float ref = dot_reference(v0.data(), v1.data(), len);
    printf("ref = %f \n", ref);
    float act = dot(v0.data(), v1.data(), len);
    printf("act = %f \n", act);

}
