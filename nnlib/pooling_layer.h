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

#ifndef POOLINGLAYER_H
#define POOLINGLAYER_H

#include "layer.h"

class PoolingLayer : public Layer
{

public:
    enum PoolingMethod {pool_max, pool_ave};
private:

    size_t kernel_size;
    size_t stride_size;
    PoolingMethod pooling_method;
    int ceiled_h;
    int ceiled_w;

public:

    PoolingLayer();

    virtual void bind(const std::vector<std::vector<size_t>> &shapes);
    virtual void forward();

    void set_kernel_size(int sz)
    {
        kernel_size = sz;
    }

    void set_stride_size(int sz)
    {
        stride_size = sz;
    }

    void set_pooling_method(PoolingMethod method)
    {
        pooling_method = method;
    }

};

#endif // POOLINGLAYER_H
