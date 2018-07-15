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

#ifndef LAYER_H
#define LAYER_H

#include <list>
#include <string>
#include <memory>
#include "ndarray.h"


class Layer
{

public:

    enum Type {
        op_fc,
        op_convolution,
        op_relu,
        op_batchnorm,
        op_scale,
        op_eltwise_sum,
        op_softmax,
        op_pool,
        op_input,
        num_types
    };

    Type op_type;
    bool inplace_possible;
    bool can_do_relu;
    bool relu;

    std::string name;
    std::string output_name;
    std::list<Layer*> input_layers;
    std::list<Layer*> output_layers;

    std::vector<std::shared_ptr<ndarray>> input_arrays;
    std::shared_ptr<ndarray> output_array;

    std::vector<std::vector<size_t>> input_shapes;
    std::vector<size_t> output_shape;


    Layer()
    {
        op_type = num_types;
        inplace_possible = false;
        can_do_relu = false;
        relu = false;
    }

    virtual ~Layer()
    {
    }

    void add_input(Layer *layer)
    {
        input_layers.push_back(layer);
    }

    void add_output(Layer *layer)
    {
        output_layers.push_back(layer);
    }

    virtual void forward()
    {

    }

    virtual void bind(const std::vector<std::vector<size_t>> &shapes)
    {
        input_shapes = shapes;
        output_shape = shapes[0];
        log_bind();
    }

    virtual void bind() //inpulayer has a different bind
    {
        std::vector<std::vector<size_t>> shapes;
        for(Layer *l : input_layers) {
            shapes.push_back(l->output_shape);
        }
        bind(shapes);
    }

    void log_bind();

};

#endif // LAYER_H
