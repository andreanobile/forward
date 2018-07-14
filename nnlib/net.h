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

#ifndef NET_H
#define NET_H

#include <vector>
#include <map>
#include <set>
#include <string>
#include <memory>

#include "layer.h"
#include "fc_layer.h"
#include "softmax_layer.h"
#include "input_layer.h"
#include "scale_layer.h"
#include "conv_layer.h"
#include "batchnorm_layer.h"
#include "rename_table.h"
#include "network_node.h"
#include "aligned_buffer.h"



class Net
{
    RenameTable rename_table;
    std::map<std::string, Layer*> out_name_to_layer;
    std::list<std::shared_ptr<Layer>> layers;
    std::vector<Layer*> vsched;
    AlignedBuffer<float, 64> fmap_buffer;
    AlignedBuffer<float, 64> conv_pad_buffer;
    AlignedBuffer<float, 64> conv_im2col_buffer;

    std::vector<Layer*> input_layers;
    std::vector<Layer*> output_layers;

    ndarray *get_input_array()
    {
        return input_layers[0]->output_array.get();
    }

    void remove_layer(const std::shared_ptr<Layer> &layer);

public:


    Net()
    {
    }

    ~Net()
    {
    }

    void forward();
    void complete_construction();
    void schedule();
    void optimize();
    bool bind(const std::vector<size_t> &shape);
    void add_layer(const std::shared_ptr<Layer> &layer);
    Layer* add_layer(Layer::Type layer_type,
                                     const NetworkNode &params,
                                     const std::vector<std::string> &inputs,
                                     const std::string &layer_name,
                                     const std::string &output_name);

    Layer* output_name_to_layer(const std::string &str)
    {
        auto it = out_name_to_layer.find(str);
        if (it != out_name_to_layer.end()) {
            return it->second;
        }
        else {
            std::cout << "item " << str << "not found in output names" << std::endl;
            std::abort();
        }
    }

    void copy_to_input(ndarray &inp)
    {
        get_input_array()->copy_from(inp);
    }

    ndarray *get_output()
    {
        return output_layers[0]->output_array.get();;
    }

    void copy_net_sharing_weights(Net &original);

};

#endif // NET_H
