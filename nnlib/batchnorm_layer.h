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

#ifndef BATCHNORMLAYER_H
#define BATCHNORMLAYER_H

#include "layer.h"
#include "scale_layer.h"

class BatchNormLayer : public Layer
{
    size_t ninner;
    float scale;
    float eps;
    bool precompute_done;

    std::shared_ptr<ndarray> mean;
    std::shared_ptr<ndarray> variance;
    std::shared_ptr<ScaleLayer> scale_layer;

    void precompute();

public:

    BatchNormLayer();

    virtual void bind(const std::vector<std::vector<size_t>> &shapes);
    virtual void forward();

    void set_mean(std::shared_ptr<ndarray> p)
    {
        mean = p;
        precompute_done = false;
    }

    void set_variance(std::shared_ptr<ndarray> p)
    {
        variance = p;
        precompute_done = false;
    }

    void set_scale(float sc)
    {
        scale = sc;
        precompute_done = false;
    }

    void set_scale_layer(std::shared_ptr<ScaleLayer> scl)
    {
        scale_layer = scl;
        precompute_done = false;
    }

};

#endif // BATCHNORMLAYER_H
