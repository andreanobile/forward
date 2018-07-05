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
#include <algorithm>
#include "pooling_layer.h"

using namespace std;


PoolingLayer::PoolingLayer()
{
    op_type = op_pool;
    inplace_possible = false;
    pooling_method = pool_max;
    kernel_size = 1;
    stride_size = 1;
    ceiled_h = 0;
    ceiled_w = 0;
}


static bool good_shape(int w, int kern_size, int stride_size)
{
    int c = ceil((w - kern_size)/(float)stride_size) + 1;
    int norm = (w - kern_size)/stride_size + 1;
    return (c == norm);
}


void PoolingLayer::bind(const vector<vector<size_t>> &shapes)
{
    input_shapes = shapes;
    output_shape = shapes[0];
    output_shape[2] = ceil((input_shapes[0][2] - kernel_size)/(float)stride_size) + 1;
    output_shape[3] = ceil((input_shapes[0][3] - kernel_size)/(float)stride_size) + 1;

    log_bind();

    if(!good_shape((input_shapes[0][2]), kernel_size, stride_size)) {
        ceiled_h = 1;
        cout << "warning " << name << " shape h has been padded" << endl;
    }
    if(!good_shape((input_shapes[0][3]), kernel_size, stride_size)) {
        ceiled_w = 1;
        cout << "warning " << name << " shape w has been padded" << endl;
    }

}


void PoolingLayer::forward()
{

    ndarray *input = input_arrays[0].get();
    ndarray *output = output_array.get();

    assert(input->shape == input_shapes[0]);
    assert(output->shape == output_shape);

    auto &inp_shape = input_shapes[0];
    size_t nch = inp_shape[1];
    size_t nbatch = inp_shape[0];
    size_t nh = inp_shape[2];
    size_t nw = inp_shape[3];

    float *ibuf = input->get_data();
    float *obuf = output->get_data();


    size_t out_h = output_shape[2];
    size_t out_w = output_shape[3];

    size_t k=0;
    for(size_t ib=0;ib<nbatch;ib++) {
        size_t boffs = ib*(nch*nh*nw);
        for(size_t ich=0;ich<nch;ich++) {
            size_t choffs = ich*(nh*nw);
            for(size_t ph=0;ph<out_h;ph++) {
                for(size_t pw=0;pw<out_w;pw++) {

                    int hstart = ph * stride_size;
                    int wstart = pw * stride_size ;
                    int hend = min(hstart + kernel_size, nh);
                    int wend = min(wstart + kernel_size, nw);

                    if(pooling_method == pool_max) {

                        float mx = -1e20f;
                        for (int h = hstart; h < hend; h++) {
                            for (int w = wstart; w < wend; w++) {
                                int index = h*nw + w;
                                float val = ibuf[boffs+choffs+index];
                                mx = max(mx, val);

                            }
                        }
                        obuf[k]=mx;
                        k++;

                    } else if(pooling_method == pool_ave) {

                        float mx = 0.0f;
                        float pool_size = (hend - hstart) * (wend - wstart);
                        for (int h = hstart; h < hend; h++) {
                            for (int w = wstart; w < wend; w++) {
                                int index = h*nw + w;
                                float val = ibuf[boffs+choffs+index];
                                mx += val;

                            }
                        }
                        obuf[k]=mx/pool_size;
                        k++;
                    }


                }
            }
        }
    }

    assert(k==output_array->n_elements());
}
