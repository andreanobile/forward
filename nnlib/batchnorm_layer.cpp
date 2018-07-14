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

#include <iostream>
#include <math.h>
#include "batchnorm_layer.h"

using namespace std;

BatchNormLayer::BatchNormLayer()
{
    op_type = op_batchnorm;
    inplace_possible = true;
    ninner = 0;
    scale = 0.0f;
    eps = 1e-5;
    precompute_done = false;
    scale_layer = nullptr;
    can_do_relu = true;
}


void BatchNormLayer::bind(const vector<vector<size_t>> &shapes)
{
    input_shapes = shapes;
    output_shape = shapes[0];

    auto &inp_shape = input_shapes[0];

    ninner = 1;
    for(size_t i=2;i<inp_shape.size();i++) {
        ninner *= inp_shape[i];
    }

    if(!precompute_done) precompute();

    log_bind();
}


static inline void scale_array(float * __restrict__ arr, size_t n, float s)
{
    for(size_t i=0;i<n;i++) {
        arr[i] *= s;
    }
}


void BatchNormLayer::precompute()
{
    size_t n = mean->shape[0];
    assert(n == variance->shape[0]);

    if(scale != 1.0f) {
        if(scale != 0.0f) {
            scale = 1.0f/scale;
        }
        scale_array(mean->get_data(), n, scale);
        scale_array(variance->get_data(), n, scale);
    }

    // variance <- 1/sqrt(variance+eps)
    float *__restrict__ invv = variance->get_data();
    for(size_t i=0;i<n;i++) {
        invv[i] = 1.0f/sqrtf(invv[i] + eps);
    }

    //(x_i - m)*invv*gamma + b
    //bt = b/(invv*gamma)
    //(x_i - m + bt)*invv*gamma
    //mt = m - bt
    //(x_i - mt)*invv*gamma
    ScaleLayer * __restrict__ sc = scale_layer.get();

    if(sc) {
        float * __restrict__ gamma = sc->weights->get_data();
        float * __restrict__ b = sc->bias->get_data();
        float * __restrict__ m = mean->get_data();

        for(size_t i=0;i<n;i++) {
            //invv <- invv*gamma
            float invvgamma = invv[i]*gamma[i];
            invv[i] = invvgamma;

            float bt = b[i]/invvgamma;

            //m <- mt
            m[i] = m[i]-bt;
        }
    }

    scale_layer = nullptr;
    precompute_done = true;
}


static inline void bn_arrays(float * __restrict__ data_in, float * __restrict__ data_out, size_t n, float nrm, float mn_nrm, bool relu)
{
    if(relu) {
        for(size_t k=0;k<n;k++) {
            //data_out[k] = (data_in[k] - mn)*nrm;
            data_out[k] = max(data_in[k]*nrm - mn_nrm, 0.0f);
        }
    } else {
        for(size_t k=0;k<n;k++) {
            //data_out[k] = (data_in[k] - mn)*nrm;
            data_out[k] = data_in[k]*nrm - mn_nrm;
        }
    }
}


void BatchNormLayer::forward()
{
    ndarray *input = input_arrays[0].get();
    ndarray *output = output_array.get();
    size_t nch = mean->shape[0];

    assert(input->shape[1] == nch);
    assert(output->shape[1] == nch);

    size_t nbatch = input->shape[0];

    float * __restrict__ din = input->get_data();
    float * __restrict__ dout = output->get_data();
    float * __restrict__ pnrm = variance->get_data();
    float * __restrict__ pmn = mean->get_data();


    for(size_t i=0;i<nbatch;i++) {
        size_t boffs = i*nch*ninner;
        for(size_t j=0;j<nch;j++) {
            size_t offs = boffs + j*ninner;

            float * __restrict__ data_in = din + offs;
            float * __restrict__ data_out = dout + offs;

            float nrm = pnrm[j];
            float mn = pmn[j];

            float mn_nrm = mn*nrm;
            bn_arrays(data_in, data_out, ninner, nrm, mn_nrm, relu);
        }
    }
}
