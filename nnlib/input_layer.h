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

#ifndef INPUT_LAYER_H
#define INPUT_LAYER_H

#include "layer.h"
#include <vector>

//using namespace std;

class InputLayer : public Layer
{
public:
    InputLayer();
    std::vector<size_t> input_dims;

    void add_dim(size_t sz)
    {
        input_dims.push_back(sz);
    }

    void bind()
    {
        output_shape = input_dims;
        log_bind();
    }

};

#endif // INPUT_LAYER_H
