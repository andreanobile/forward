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
#include <iostream>
#include "softmax_layer.h"
#include <math.h>

using namespace std;


SoftmaxLayer::SoftmaxLayer()
{
    op_type = op_softmax;
    inplace_possible = true;
    ninner = 0;
}


void SoftmaxLayer::bind(const vector<vector<size_t>> &shapes)
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


void SoftmaxLayer::forward()
{
    //cout << "softmax " << name << " forward " << endl;

    ndarray *input = input_arrays[0].get();
    ndarray *output = output_array.get();

    assert(input->shape == input_shapes[0]);
    assert(output->shape == output_shape);

    auto &inp_shape = input_shapes[0];
    size_t nch = inp_shape[1];
    size_t nbatch = inp_shape[0];

#ifndef NDEBUG
    size_t sz = 1;
    for(auto i: inp_shape) {
        sz *= i;
    }
#endif

    float *odata = output->get_data();
    float *idata = input->get_data();

    if(ninner == 1) {
        for(size_t j=0;j<nbatch;j++) {
            size_t boffs = j*nch;

            float s = 0.0f;
            for(size_t k=0;k<nch;k++) {
                size_t offs = boffs+k;
                assert(offs < sz);
                float val = expf(idata[offs]);
                s += val;
                odata[offs] = val;
            }
            s = 1.0f/s;
            for(size_t k=0;k<nch;k++) {
                size_t offs = boffs+k;
                float v = odata[offs];
                v = v*s;
                odata[offs] = v;
            }
        }
    } else {
        for(size_t j=0;j<nbatch;j++) {
            size_t boffs = j*nch*ninner;
            for(size_t i=0;i<ninner;i++) {
                float s = 0.0f;
                for(size_t k=0;k<nch;k++) {
                    size_t offs = boffs+k*ninner + i;
                    assert(offs < sz);
                    float val = expf(idata[offs]);
                    s += val;
                    odata[offs] = val;
                }
                s = 1.0f/s;
                for(size_t k=0;k<nch;k++) {
                    size_t offs = boffs+k*ninner + i;
                    float v = odata[offs];
                    v = v*s;
                    odata[offs] = v;
                }
            }
        }
    }

}
