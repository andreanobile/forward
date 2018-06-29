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
#include "eltwise_layer.h"

using namespace std;


EltWiseLayer::EltWiseLayer()
{
    op_type = op_eltwise_sum;
    inplace_possible = false;
    ninner = 0;
    can_do_relu = true;
    relu = false;
}


void EltWiseLayer::bind(const vector<vector<size_t>> &shapes)
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


static inline void add_arrays(float * __restrict__ in0, float * __restrict__ in1,
                              float * __restrict__ out, size_t n, bool do_relu)
{
    if(do_relu) {
        for(size_t i=0;i<n;i++) {
            out[i] = max(in0[i]+in1[i], 0.0f);
        }
    } else {
        for(size_t i=0;i<n;i++) {
            out[i] = in0[i]+in1[i];
        }
    }
}


void EltWiseLayer::forward()
{
    //cout << "eltwise_sum " << name << " forward" << endl;

    size_t nch;
    ndarray *input0 = input_arrays[0].get();
    ndarray *input1 = input_arrays[1].get();
    ndarray *output = output_array.get();
    nch = input0->shape[1];

    assert(input1->shape[1] == nch);
    assert(output->shape[1] == nch);

    size_t nbatch = input0->shape[0];
    size_t sz = nbatch*nch*ninner;

    float *data_in0 = input0->get_data();
    float *data_in1 = input1->get_data();
    float *data_out = output->get_data();

    add_arrays(data_in0, data_in1, data_out, sz, relu);
}
