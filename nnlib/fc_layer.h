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

#ifndef FC_LAYER_H
#define FC_LAYER_H

#include "layer.h"
#include "ndarray.h"
#include <vector>
#include <memory>

class FCLayer : public Layer
{

    std::shared_ptr<ndarray> weights;
    std::shared_ptr<ndarray> bias;
    size_t num_output_channels;
    size_t num_input_channels;


public:


    FCLayer()
    {
        init();
    }

    explicit FCLayer(size_t nout_channels)
    {
        init();
        set_num_output_channels(nout_channels);
    }

    void init();

    virtual void bind(const std::vector<std::vector<size_t>> &shapes);

    virtual void forward();

    void set_weights(std::shared_ptr<ndarray> p)
    {
        weights = p;
    }
    void set_bias(std::shared_ptr<ndarray> p)
    {
        bias = p;
    }

    void set_num_output_channels(size_t nc)
    {
        num_output_channels = nc;
    }

    void setnum_input_channels(size_t nc)
    {
        num_input_channels = nc;
    }

};

#endif // FC_LAYER_H
