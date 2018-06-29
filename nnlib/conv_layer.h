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

#ifndef CONVLAYER_H
#define CONVLAYER_H

#include <memory>
#include "layer.h"


class ConvLayer : public Layer
{
    std::shared_ptr<ndarray> weights;
    std::shared_ptr<ndarray> bias;

    size_t num_output_channels;
    size_t num_input_channels;
    size_t kernel_size;
    size_t pad_size;
    size_t stride_size;
    bool precompute_done;

    void im2col(ndarray *im, ndarray *result, size_t ksz, size_t stride);

public:
    bool has_bias;

    ConvLayer();

    virtual void bind(const std::vector<std::vector<size_t>> &shapes);
    virtual void forward();

    void set_weights(std::shared_ptr<ndarray> p)
    {
        weights = p;
        precompute_done = false;
    }
    void set_bias(std::shared_ptr<ndarray> p)
    {
        bias = p;
        has_bias = true;
        precompute_done = false;
    }

    void set_num_output_channels(size_t nc)
    {
        num_output_channels = nc;
    }

    size_t get_num_output_channels()
    {
        return num_output_channels;
    }

    void set_num_input_channels(size_t nc)
    {
        num_input_channels = nc;
    }

    size_t get_num_input_channels()
    {
        return num_input_channels;
    }

    void set_pad_size(int sz)
    {
        pad_size = sz;
    }

    void set_kernel_size(int sz)
    {
        kernel_size = sz;
    }

    void set_stride_size(int sz)
    {
        stride_size = sz;
    }

};

#endif // CONVLAYER_H
