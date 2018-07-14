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

#include "fc_layer.h"
#include <assert.h>
#include "kernels.h"

#define RELU(x) ((x) >= 0.0f) ? (x) : 0.0f

using namespace std;


void FCLayer::init()
{
    op_type = op_fc;
    inplace_possible = false;
    num_output_channels = 0;
    num_input_channels = 0;
    can_do_relu = true;
}


void FCLayer::bind(const vector<vector<size_t>> &shapes)
{
    assert(shapes.size()==1);
    const vector<size_t> &s = shapes[0];
    num_input_channels = s[1];

    // num output channels is set during creation
    input_shapes = shapes;
    output_shape = shapes[0];
    output_shape[1] = num_output_channels;
    bias->reshape({num_output_channels});
    weights->reshape({num_output_channels, num_input_channels});

    log_bind();
}


void FCLayer::forward()
{
    //cout << "fc " << name << " forward" << endl;

    size_t n_in, n_out;

    n_out = weights->shape[0];
    n_in = weights->shape[1];

    ndarray *input = input_arrays[0].get();
    ndarray *output = output_array.get();

    //check that data shape matches with fc layer shape
    //FIXME: NOT OK for batches
    assert(input->shape[1] == n_in);
    assert(output->shape[1] == n_out);
    assert(n_out == num_output_channels);

    matvec(weights->get_data(), input->get_data(), output->get_data(), n_out, n_in);

    //add bias  and relu

    size_t i;
    float *d = output->get_data();
    float *bdata = bias->get_data();

    if(relu) {
        for(i=0;i<n_out;i++) {
            d[i] = RELU(d[i] + bdata[i]);
        }
    } else {
        for(i=0;i<n_out;i++) {
            d[i] = d[i] + bdata[i];
        }
    }
}
