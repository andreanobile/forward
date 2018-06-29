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

#ifndef SCALELAYER_H
#define SCALELAYER_H

#include "layer.h"
#include <memory>

class ScaleLayer : public Layer
{

    size_t ninner;

public:
    std::shared_ptr<ndarray> weights;
    std::shared_ptr<ndarray> bias;
    bool has_bias;

    ScaleLayer();

    void set_weights(std::shared_ptr<ndarray> p)
    {
        weights = p;
    }


    void set_bias(std::shared_ptr<ndarray> p)
    {
        bias = p;
        has_bias = true;
    }


    virtual void bind(const std::vector<std::vector<size_t>> &shapes);
    virtual void forward();
};

#endif // SCALELAYER_H
