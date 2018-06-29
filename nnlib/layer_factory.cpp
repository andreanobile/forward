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


shared_ptr<Layer> create_layer(Layer::Type op_type, shared_ptr<NetworkNode> params)
{
    shared_ptr<Layer> layer;


    if(op_type == Layer::op_convolution)
    {
        shared_ptr<Layer> ptr(new ConvLayer());
        auto conv = static_cast<ConvLayer*>(ptr.get());
        auto vs = params->get_prop("num_output");
        assert(vs.size() == 1);
        size_t nout = stoi(vs[0]);
        conv->set_num_output_channels(nout);

        vs = params->get_prop("kernel_size");
        assert(vs.size() == 1);
        conv->set_kernel_size(stoi(vs[0]));

        vs = params->get_prop("pad");
        assert(vs.size() == 1);
        conv->set_pad_size(stoi(vs[0]));

        vs = params->get_prop("stride");
        assert(vs.size() == 1);
        conv->set_stride_size(stoi(vs[0]));

        vs = params->get_prop("bias_term");
        if(vs.size() != 0) {
            if(vs[0] == "false") {
                conv->has_bias = false;
            } else {
                conv->has_bias = true;
            }
        }

        return ptr;
    }

    else if(op_type == Layer::op_batchnorm)
    {
        shared_ptr<Layer> ptr(new BatchNormLayer());
        return ptr;
    }

    else if(op_type == Layer::op_softmax)
    {
        shared_ptr<Layer> ptr(new SoftmaxLayer());
        return ptr;
    }

    else if(op_type == Layer::op_fc)
    {
        shared_ptr<Layer> ptr(new FCLayer());
        auto fc = static_cast<FCLayer*>(ptr.get());
        auto vs = params->get_prop("num_output");
        assert(vs.size() == 1);
        size_t nout = stoi(vs[0]);
        fc->set_num_output_channels(nout);
        return ptr;
    }

    else if(op_type == Layer::op_relu)
    {
        shared_ptr<Layer> ptr(new ReLULayer());
        return ptr;
    }

    else if(op_type == Layer::op_scale)
    {
        shared_ptr<Layer> ptr(new ScaleLayer());
        auto sc = static_cast<ScaleLayer*>(ptr.get());
        auto vs = params->get_prop("bias_term");
        if(vs.size()) {
            if(vs[0] == "false") {
                sc->has_bias = false;
            } else {
                sc->has_bias = true;
            }
        }
        return ptr;
    }

    else if(op_type == Layer::op_pool)
    {
        shared_ptr<Layer> ptr(new PoolingLayer());
        auto pool = static_cast<PoolingLayer*>(ptr.get());
        auto vs = params->get_prop("kernel_size");
        assert(vs.size() == 1);
        pool->set_kernel_size(stoi(vs[0]));

        vs = params->get_prop("stride");
        assert(vs.size() == 1);
        pool->set_stride_size(stoi(vs[0]));

        vs = params->get_prop("pool");
        if(vs.size() == 1){
            pool->set_pooling_method(vs[0] == "AVE" ? pool->pool_ave : pool->pool_max);
        }

        return ptr;
    }

    else if(op_type == Layer::op_eltwise_sum)
    {
        shared_ptr<Layer> ptr(new EltWiseLayer());
        return ptr;
    }

    else if(op_type == Layer::op_input)
    {
        shared_ptr<Layer> ptr(new InputLayer());
        auto inp = static_cast<InputLayer*>(ptr.get());
        auto p = params->get_prop("input_dim");
        for (auto &d : p) {
            inp->add_dim(stoi(d));
        }
        return ptr;
    }

    else
    {
        cout << "passed invalid op_type to create_layer" << endl;
        abort();
        return layer;
    }
}

template<typename T>
static void copylayer(Layer *n, Layer*o)
{
    auto nt = static_cast<T*>(n);
    auto ot = static_cast<T*>(o);
    *nt = *ot;
}

shared_ptr<Layer> copy_layer(Layer *l)
{

    shared_ptr<Layer> layer;
    Layer::Type op_type = l->op_type;

    if(op_type == Layer::op_convolution)
    {
        shared_ptr<Layer> ptr(new ConvLayer());
        copylayer<ConvLayer>(ptr.get(), l);
        return ptr;
    }

    else if(op_type == Layer::op_batchnorm)
    {
        shared_ptr<Layer> ptr(new BatchNormLayer());
        copylayer<BatchNormLayer>(ptr.get(), l);
        return ptr;
    }

    else if(op_type == Layer::op_softmax)
    {
        shared_ptr<Layer> ptr(new SoftmaxLayer());
        copylayer<SoftmaxLayer>(ptr.get(), l);
        return ptr;
    }

    else if(op_type == Layer::op_fc)
    {
        shared_ptr<Layer> ptr(new FCLayer());
        copylayer<FCLayer>(ptr.get(), l);
        return ptr;
    }

    else if(op_type == Layer::op_relu)
    {
        shared_ptr<Layer> ptr(new ReLULayer());
        copylayer<ReLULayer>(ptr.get(), l);
        return ptr;
    }

    else if(op_type == Layer::op_scale)
    {
        shared_ptr<Layer> ptr(new ScaleLayer());
        copylayer<ScaleLayer>(ptr.get(), l);
        return ptr;
    }

    else if(op_type == Layer::op_pool)
    {
        shared_ptr<Layer> ptr(new PoolingLayer());
        copylayer<PoolingLayer>(ptr.get(), l);
        return ptr;
    }

    else if(op_type == Layer::op_eltwise_sum)
    {
        shared_ptr<Layer> ptr(new EltWiseLayer());
        copylayer<EltWiseLayer>(ptr.get(), l);
        return ptr;
    }

    else if(op_type == Layer::op_input)
    {
        shared_ptr<Layer> ptr(new InputLayer());
        copylayer<InputLayer>(ptr.get(), l);
        return ptr;
    }

    else
    {
        cout << "passed invalid op_type to create_layer" << endl;
        abort();
        return layer;
    }
}

