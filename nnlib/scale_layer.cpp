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

#include <algorithm>
#include <assert.h>
#include "scale_layer.h"

using namespace std;


ScaleLayer::ScaleLayer()
{
    op_type = op_scale;
    has_bias = false;
    inplace_possible = true;
    ninner = 0;
    can_do_relu = true;
}


void ScaleLayer::bind(const vector<vector<size_t>> &shapes)
{
    input_shapes = shapes;
    output_shape = shapes[0];

    auto &inp_shape = input_shapes[0];

    ninner = 1;
    for(size_t i=2;i<inp_shape.size();i++) {
        ninner *= inp_shape[i];
    }

    log_bind();
}


static inline void scale_with_bias(const float * __restrict__ data_in, float * __restrict__ data_out,
                                   size_t n, float scale, float bias, bool relu)
{
    if(relu) {
        for(size_t k=0;k<n;k++) {
            data_out[k] = max(scale*data_in[k] + bias, 0.0f);
        }
    } else {
        for(size_t k=0;k<n;k++) {
            data_out[k] = scale*data_in[k] + bias;
        }
    }
}


static inline void scale(const float * __restrict__ data_in, float * __restrict__ data_out,
                         size_t n, float scale, bool relu)
{
    if(relu) {
        for(size_t k=0;k<n;k++) {
            data_out[k] = max(scale*data_in[k], 0.0f);
        }
    } else {
        for(size_t k=0;k<n;k++) {
            data_out[k] = scale*data_in[k];
        }
    }
}


void ScaleLayer::forward()
{
    size_t nch;
    ndarray *input = input_arrays[0].get();
    ndarray *output = output_array.get();
    nch = weights->shape[0];

    //check that data shape matches with layer shape
    assert(input->shape[1] == nch);
    assert(output->shape[1] == nch);
    size_t nbatch = input->shape[0];

    for(size_t i=0;i<nbatch;i++) {
        size_t boffs = i*nch*ninner;
        for(size_t j=0;j<nch;j++) {
            size_t offs = boffs + j*ninner;

            float scale_param = weights->get_data()[j];
            float *data_in = &input->get_data()[offs];
            float *data_out = &output->get_data()[offs];

            if(has_bias) {
                float bias_param = bias->get_data()[j];
                scale_with_bias(data_in, data_out, ninner, scale_param, bias_param, relu);
            } else {
                scale(data_in, data_out, ninner, scale_param, relu);
            }

        }
    }
}
