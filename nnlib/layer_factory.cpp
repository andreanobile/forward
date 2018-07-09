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

#include <memory>
#include <utility>
#include <iostream>
#include <assert.h>
#include "layer_factory.h"
#include "layer.h"
#include "fc_layer.h"
#include "softmax_layer.h"
#include "relu_layer.h"
#include "batchnorm_layer.h"
#include "scale_layer.h"
#include "conv_layer.h"
#include "pooling_layer.h"
#include "eltwise_layer.h"
#include "input_layer.h"

using namespace std;


shared_ptr<Layer> create_layer(Layer::Type op_type, const NetworkNode &params)
{

    if(op_type == Layer::op_convolution)
    {
        auto conv = make_shared<ConvLayer>();
        auto vs = params.get_prop("num_output");
        assert(vs.size() == 1);
        size_t nout = stoi(vs[0]);
        conv->set_num_output_channels(nout);

        vs = params.get_prop("kernel_size");
        assert(vs.size() == 1);
        conv->set_kernel_size(stoi(vs[0]));

        vs = params.get_prop("pad");
        assert(vs.size() == 1);
        conv->set_pad_size(stoi(vs[0]));

        vs = params.get_prop("stride");
        assert(vs.size() == 1);
        conv->set_stride_size(stoi(vs[0]));

        vs = params.get_prop("bias_term");
        if(vs.size() != 0) {
            if(vs[0] == "false") {
                conv->has_bias = false;
            } else {
                conv->has_bias = true;
            }
        }

        return static_pointer_cast<Layer>(conv);
    }

    else if(op_type == Layer::op_batchnorm)
    {
        auto ptr = make_shared<BatchNormLayer>();
        return static_pointer_cast<Layer>(ptr);
    }

    else if(op_type == Layer::op_softmax)
    {
        auto ptr = make_shared<SoftmaxLayer>();
        return static_pointer_cast<Layer>(ptr);
    }

    else if(op_type == Layer::op_fc)
    {
        auto fc = make_shared<FCLayer>();
        auto vs = params.get_prop("num_output");
        assert(vs.size() == 1);
        size_t nout = stoi(vs[0]);
        fc->set_num_output_channels(nout);
        return static_pointer_cast<Layer>(fc);
    }

    else if(op_type == Layer::op_relu)
    {
        auto ptr = make_shared<ReLULayer>();
        return static_pointer_cast<Layer>(ptr);
    }

    else if(op_type == Layer::op_scale)
    {
        auto sc = make_shared<ScaleLayer>();
        auto vs = params.get_prop("bias_term");
        if(vs.size()) {
            if(vs[0] == "false") {
                sc->has_bias = false;
            } else {
                sc->has_bias = true;
            }
        }
        return static_pointer_cast<Layer>(sc);
    }

    else if(op_type == Layer::op_pool)
    {
        auto pool = make_shared<PoolingLayer>();
        auto vs = params.get_prop("kernel_size");
        assert(vs.size() == 1);
        pool->set_kernel_size(stoi(vs[0]));

        vs = params.get_prop("stride");
        assert(vs.size() == 1);
        pool->set_stride_size(stoi(vs[0]));

        vs = params.get_prop("pool");
        if(vs.size() == 1){
            pool->set_pooling_method(vs[0] == "AVE" ? pool->pool_ave : pool->pool_max);
        }

        return static_pointer_cast<Layer>(pool);
    }

    else if(op_type == Layer::op_eltwise_sum)
    {
        auto ptr = make_shared<EltWiseLayer>();
        return static_pointer_cast<Layer>(ptr);
    }

    else if(op_type == Layer::op_input)
    {
        auto inp = make_shared<InputLayer>();
        auto p = params.get_prop("input_dim");
        for (auto &d : p) {
            inp->add_dim(stoi(d));
        }
        return static_pointer_cast<Layer>(inp);
    }

    else
    {
        cout << "passed invalid op_type to create_layer" << endl;
        abort();
        return nullptr;
    }
}


template<typename T>
static void copylayer(Layer *n, Layer*o)
{
    auto nt = static_cast<T*>(n);
    auto ot = static_cast<T*>(o);
    *nt = *ot;
}


template<typename T>
static shared_ptr<Layer> make_and_copy(Layer *l)
{
    auto ptr = make_shared<T>();
    copylayer<T>(static_cast<Layer*>(ptr.get()), l);
    return static_pointer_cast<Layer>(ptr);
}


shared_ptr<Layer> copy_layer(Layer *l)
{

    Layer::Type op_type = l->op_type;

    if(op_type == Layer::op_convolution)
    {
        return make_and_copy<ConvLayer>(l);
    }

    else if(op_type == Layer::op_batchnorm)
    {
        return make_and_copy<BatchNormLayer>(l);
    }

    else if(op_type == Layer::op_softmax)
    {
        return make_and_copy<SoftmaxLayer>(l);
    }

    else if(op_type == Layer::op_fc)
    {
        return make_and_copy<FCLayer>(l);
    }

    else if(op_type == Layer::op_relu)
    {
        return make_and_copy<ReLULayer>(l);
    }

    else if(op_type == Layer::op_scale)
    {
        return make_and_copy<ScaleLayer>(l);
    }

    else if(op_type == Layer::op_pool)
    {
        return make_and_copy<PoolingLayer>(l);
    }

    else if(op_type == Layer::op_eltwise_sum)
    {
        return make_and_copy<EltWiseLayer>(l);
    }

    else if(op_type == Layer::op_input)
    {
        return make_and_copy<InputLayer>(l);
    }

    else
    {
        cout << "passed invalid op_type to create_layer" << endl;
        abort();
        return nullptr;
    }
}

