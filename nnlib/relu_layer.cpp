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
#include "relu_layer.h"

using namespace std;

#define RELU(x) (((x) >= 0.0f) ? (x) : 0.0f)

ReLULayer::ReLULayer()
{
    op_type = op_relu;
    inplace_possible = true;
    ntot = 0;
}

void ReLULayer::bind(const vector<vector<size_t>> &shapes)
{
    input_shapes = shapes;
    output_shape = shapes[0];

    auto &inp_shape = input_shapes[0];

    assert(inp_shape.size());

    ntot = 1;
    for(size_t i=0;i<inp_shape.size();i++) {
        ntot *= inp_shape[i];
    }


    log_bind();
}

void ReLULayer::forward()
{
    //cout << "relu " << name << " forward " << endl;

    float * __restrict__ in = input_arrays[0]->get_data();
    float * __restrict__ out = output_array->get_data();
    for(size_t i=0;i<ntot;i++) {
        out[i] = max(in[i], 0.0f);//RELU(in[i]);
    }
}
