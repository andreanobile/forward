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

#ifndef CAFFE_LOADER_H
#define CAFFE_LOADER_H

#include <string>
#include <sstream>
#include <map>
#include <stack>
#include <memory>

#include "network_node.h"
#include "net.h"
#include "layer.h"
#include "caffe_network_description.h"

class CaffeLoader
{
    std::map<std::string, Layer::Type> caffe_layer_type_to_layer_type;
    std::map<std::string, int> caffe_layer_type_to_num_data_arrays;

    std::string data_files_path;

    Layer::Type map_layer(const std::string &type);
    int num_data_arrays(Layer::Type layer_type);

    void init_caffe_layer_to_layer_type();

    std::unique_ptr<Net> build_network(const CaffeNetworkDescription &desc);
    void add_layer(Net *net, const NetworkNode &caffe_layer);
    void load_layer_data(Layer* layer, const NetworkNode &caffe_layer);

    std::vector<std::shared_ptr<ndarray>> load_data_files(const std::string &caffe_name, int nfiles);

public:

    CaffeLoader();
    std::unique_ptr<Net> load_prototxt(const std::string &fname, const std::string &data_dir);
    std::shared_ptr<ndarray> load_data(const std::string &fname);
};

#endif // CAFFE_LOADER_H
