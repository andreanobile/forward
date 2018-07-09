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
#include <assert.h>
#include <math.h>
#include "conv_layer.h"

#define MATMUL_USE_MATMAT
#define IM2COL_USE_MEMCPY

#ifdef MATMUL_USE_BLAS
#include <cblas.h>
#else
#include "matmat.h"
#endif


using namespace std;

ConvLayer::ConvLayer()
{
    inplace_possible = false;
    has_bias = true;
    op_type = op_convolution;

    num_output_channels = 0;
    num_input_channels = 0;
    kernel_size = 1;
    pad_size = 0;
    stride_size = 1;
}


void ConvLayer::bind(const std::vector<std::vector<size_t>> &shapes)
{
    input_shapes = shapes;
    output_shape = shapes[0];

    assert(output_shape[0] == 1);

    output_shape[1] = num_output_channels;
    output_shape[2] = (output_shape[2]+2*pad_size - kernel_size)/stride_size + 1;
    output_shape[3] = (output_shape[3]+2*pad_size - kernel_size)/stride_size + 1;

    set_num_input_channels(input_shapes[0][1]);

    weights->reshape({num_output_channels, num_input_channels*kernel_size*kernel_size});

    if(has_bias) {
        bias->reshape({num_output_channels});
    }

    log_bind();
}


static inline void matmul(ndarray *ma, ndarray *mb, ndarray *mc, size_t m, size_t n, size_t k)
{

#ifdef MATMUL_USE_MATMAT

    matmat(ma->get_data(), mb->get_data(), mc->get_data(), m, n, k, 1, ma->get_lock_address());

#else

#ifdef MATMUL_USE_BLAS

    //with weights in normal form
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                1.0, ma->get_data(), k, mb->get_data(), n, 0.0, mc->get_data(), n);

#else

    for(size_t i=0;i<m;i++) {
        for(size_t j=0;j<n;j++) {
            float val = 0.0;
            for(size_t kk=0;kk<k;kk++) {
                val += ma->element(i, kk)*mb->element(kk, j);
            }
            mc->element(i, j) = val;
        }
    }

#endif
#endif


}


void ConvLayer::im2col(ndarray *im, ndarray *result, size_t ksz, size_t stride)
{
    size_t i, j, ich;
    size_t nh, nw, nch;

    nch = im->shape[1];
    nh = im->shape[2];
    nw = im->shape[3];

    float * __restrict__ obuf = result->get_data();
    float * __restrict__ ibuf = im->get_data();

    size_t max_starty = nh-ksz;
    size_t max_startx = nw-ksz;

    if(stride != 1) {
        size_t k=0;
        for(ich=0;ich<nch;ich++) {
            size_t choffs = ich*(nh*nw);
            for(i=0;i<ksz;i++) {
                size_t khoffs = i*nw;
                for(j=0;j<ksz;j++) {
                    for(size_t ih=0;ih<=max_starty;ih+=stride) {
                        size_t hoffs = ih*nw;
                        for(size_t iw=0;iw<=max_startx;iw+=stride) {
                            float val = ibuf[choffs+hoffs+iw+khoffs+j];
                            obuf[k] = val;
                            k++;
                        }
                    }
                }
            }
        }
    } else {
        size_t k=0;
        for(ich=0;ich<nch;ich++) {
            size_t choffs = ich*(nh*nw);
            for(i=0;i<ksz;i++) {
                size_t khoffs = i*nw;
                for(j=0;j<ksz;j++) {
                    for(size_t ih=0;ih<=max_starty;ih++) {
                        size_t hoffs = ih*nw;
#ifdef IM2COL_USE_MEMCPY
                        memcpy(obuf+k, ibuf+choffs+hoffs+khoffs+j, sizeof(float)*(max_startx+1));
                        k+=max_startx+1;
#else
                        for(size_t iw=0;iw<=max_startx;iw++) {
                            float val = ibuf[choffs+hoffs+iw+khoffs+j];
                            obuf[k] = val;
                            k++;
                        }
#endif
                    }
                }
            }
        }
    }
}


void ConvLayer::forward()
{

    ndarray *input = input_arrays[0].get();
    ndarray *output = output_array.get();

    assert(input->shape == input_shapes[0]);
    assert(output->shape == output_shape);

    auto &inp_shape = input_shapes[0];
    size_t nch = inp_shape[1];
    assert(nch = num_input_channels);
    size_t nbatch = inp_shape[0];
    size_t nph = inp_shape[2]+2*pad_size;
    size_t npw = inp_shape[3]+2*pad_size;
    size_t nh = inp_shape[2];
    size_t nw = inp_shape[3];

    ndarray padded_input;

    float *iorig = input->get_data();
    float *ibuf;

    if(pad_size == 0) {
        ibuf = iorig;
    } else {
        padded_input.allocate({nbatch, nch, nph, npw});
        padded_input.zero();
        ibuf = padded_input.get_data();
        //copy input data into padded array
        for(size_t ib=0;ib<nbatch;ib++) {
            size_t boffs = ib*(nch*nh*nw);
            size_t bpoffs = ib*(nch*nph*npw);
            for(size_t ich=0;ich<nch;ich++) {
                size_t choffs = ich*(nh*nw);
                size_t chpoffs = ich*(nph*npw);
                size_t iph = pad_size;
                for(size_t ih=0;ih<nh;ih++) {
                    memcpy(&ibuf[bpoffs+chpoffs + iph*npw + pad_size], &iorig[boffs + choffs + ih*nw], sizeof(float)*nw);
                    iph++;
                }
            }
        }

    }


#ifndef USE_CONV_REFERENCE

    size_t m = output_shape[1];
    size_t n = output_shape[2]*output_shape[3];
    size_t k = num_input_channels*kernel_size*kernel_size;

    if(kernel_size != 1 || stride_size != 1) {

        ndarray im2col_buffer({kernel_size*kernel_size*num_input_channels,
                               output_shape[2]*output_shape[3]});
        ndarray im;
        ndarray oa;

        for(size_t ib=0;ib<output_shape[0];ib++) {
            im.attach(ibuf+ib*nch*npw*nph, {1, nch, nph, npw});
            im2col(&im, &im2col_buffer, kernel_size, stride_size);
            assert(m*n*output_shape[0] == output->n_elements());
            oa.attach(output->get_data()+ib*m*n, {m, n});
            matmul(weights.get(), &im2col_buffer, &oa, m, n, k);
        }

    } else { //kernel_size==1 and stride==1

        ndarray im;
        ndarray oa;

        for(size_t ib=0;ib<output_shape[0];ib++) {
            assert(m*n*output_shape[0] == output->n_elements());
            im.attach(ibuf+ib*nch*npw*nph, {1, nch, nph, npw});
            oa.attach(output->get_data()+ib*m*n, {m, n});
            matmul(weights.get(), &im, &oa, m, n, k);
        }
    }


    if(has_bias) {

        size_t out_idx=0;
        float * __restrict__ obuf = output->get_data();
        float * __restrict__ bdata = bias->get_data();
        size_t os_inner = output_shape[2]*output_shape[3];
        size_t noch = num_output_channels;

        for(size_t ib=0;ib<nbatch;ib++) {
            for(size_t ioch=0;ioch<noch;ioch++) {
                float bias_w = bdata[ioch];
                for(size_t i=0;i<os_inner;i++) {
                    obuf[out_idx] += bias_w;
                    out_idx++;
                }
            }
        }
    }


#else

    size_t max_starty = nph-kernel_size;
    size_t max_startx = npw-kernel_size;

    float *obuf = output->get_data();
    float *wdata = weights->get_data();

    size_t k=0;
    for(size_t ib=0;ib<nbatch;ib++) {
        size_t boffs = ib*(nch*nph*npw);
        for(size_t ioch=0;ioch<num_output_channels;ioch++) {
            size_t kochoffs = ioch*(kernel_size*kernel_size*num_input_channels);

            float bias_w = 0.0f;
            if(has_bias) {
                bias_w = bias->get_data()[ioch];
            }
            size_t ppp=0;
            for(size_t ih=0;ih<=max_starty;ih+=stride_size) {
                size_t hoffs = ih*npw;
                for(size_t iw=0;iw<=max_startx;iw+=stride_size) {

                    float res;
                    res = 0.0f;

                    for(size_t ich=0;ich<nch;ich++) {
                        size_t choffs = ich*(nph*npw);


                        float *patch = &ibuf[boffs+choffs+hoffs+iw];
                        size_t is=0;
                        size_t koffs = kochoffs + ich*kernel_size*kernel_size;

                        for(size_t i=0;i<kernel_size;i++) {
                            size_t khoffs = i*npw;
                            for(size_t j=0;j<kernel_size;j++) {

                                float val = patch[khoffs+j];
                                res += val*wdata[koffs + is];
                                is++;
                            }
                        }

                    }

                    res += bias_w;
                    obuf[k]=res;
                    k++;
                    ppp++;

                }
            }


        }
    }

    assert(k==output->n_elements());
#endif
}
