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

#include <iostream>
#include <fstream>
#include <cstdio>
#include <string>
#include <vector>
#include <sstream>
#include <map>
#include <stack>
#include <memory>
#include "caffe_loader.h"
#include "net.h"
#include "string_utils.h"
#include "file_utils.h"
#include "ndarray.h"
#include "caffe_network_description.h"

using namespace std;


CaffeLoader::CaffeLoader()
{
    init_caffe_layer_to_layer_type();
}


void CaffeLoader::init_caffe_layer_to_layer_type()
{
    caffe_layer_type_to_layer_type.insert({"BatchNorm", Layer::op_batchnorm});
    caffe_layer_type_to_layer_type.insert({"InnerProduct", Layer::op_fc});
    caffe_layer_type_to_layer_type.insert({"Convolution", Layer::op_convolution});
    caffe_layer_type_to_layer_type.insert({"Scale", Layer::op_scale});
    caffe_layer_type_to_layer_type.insert({"ReLU", Layer::op_relu});
    caffe_layer_type_to_layer_type.insert({"Pooling", Layer::op_pool});
    caffe_layer_type_to_layer_type.insert({"Eltwise", Layer::op_eltwise_sum});
    caffe_layer_type_to_layer_type.insert({"Softmax", Layer::op_softmax});
}


Layer::Type CaffeLoader::map_layer(const std::string &type)
{
    return caffe_layer_type_to_layer_type[type];
}


int CaffeLoader::num_data_arrays(Layer::Type layer_type)
{

    switch(layer_type) {

    case Layer::op_convolution:
        return 2;
        break;
    case Layer::op_batchnorm:
        return 3;
        break;
    case Layer::op_scale:
        return 1;
        break;
    case Layer::op_fc:
        return 2;
        break;
    case Layer::op_softmax:
        return 0;
        break;
    case Layer::op_relu:
        return 0;
        break;
    case Layer::op_pool:
        return 0;
        break;
    case Layer::op_eltwise_sum:
        return 0;
        break;

    default:
        cout << "unknown layer type " << endl;
        abort();
        return 0;

    }
}


vector<shared_ptr<ndarray>> CaffeLoader::load_data_files(const string &caffe_name, int nfiles)
{
    const string &path = data_files_path;
    vector<shared_ptr<ndarray>> data_arrays;

    for(int i=0;i<nfiles;i++) {
        string fname = caffe_name + "_" + to_string(i) + ".dat";
        data_arrays.push_back(ndarray_from_file(path_join(path, fname)));
    }

    return data_arrays;
}


void CaffeLoader::load_layer_data(Layer* layer, const NetworkNode &caffe_layer)
{
    vector<shared_ptr<ndarray>> data_arrays;
    const string &caffe_name = caffe_layer.properties.find("name")->second;
    Layer::Type layer_type = layer->op_type;

    int nfiles = num_data_arrays(layer_type);

    if(layer_type == Layer::op_convolution) {
        ConvLayer *conv = static_cast<ConvLayer*>(layer);
        data_arrays = load_data_files(caffe_name, nfiles - (conv->has_bias ? 0 : 1));
        conv->set_weights(data_arrays[0]);
        if(conv->has_bias) {
            conv->set_bias(data_arrays[1]);
        }
    }

    else if(layer_type == Layer::op_scale) {
        ScaleLayer *sc = static_cast<ScaleLayer*>(layer);
        data_arrays = load_data_files(caffe_name, nfiles + (sc->has_bias ? 1 : 0));
        sc->set_weights(data_arrays[0]);
        if(sc->has_bias) {
            sc->set_bias(data_arrays[1]);
        }
    }

    else if(layer_type == Layer::op_batchnorm) {
        BatchNormLayer *bn = static_cast<BatchNormLayer*>(layer);
        data_arrays = load_data_files(caffe_name, nfiles);
        bn->set_scale(data_arrays[2]->get_data()[0]);
        bn->set_mean(data_arrays[0]);
        bn->set_variance(data_arrays[1]);
    }

    else if(layer_type == Layer::op_fc) {
        data_arrays = load_data_files(caffe_name, nfiles);
        FCLayer *fc = static_cast<FCLayer*>(layer);
        fc->set_weights(data_arrays[0]);
        fc->set_bias(data_arrays[1]);
    }
}


void CaffeLoader::add_layer(Net *net, const NetworkNode &caffe_layer)
{
    auto it = caffe_layer.properties.find("name");
    if (it != caffe_layer.properties.end()) {
        auto caffe_layer_type = caffe_layer.properties.find("type");

        Layer::Type layer_type = map_layer(caffe_layer_type->second);

        auto bottom_layers = caffe_layer.properties.equal_range("bottom");

        vector<string> vinputs;
        for(auto bt=bottom_layers.first; bt != bottom_layers.second; bt++ ) {
            vinputs.push_back(bt->second);
        }

        auto top_layers = caffe_layer.properties.equal_range("top");
        if(caffe_layer.properties.count("top") > 1) {
            cout << "wrong assumption in number of tops at layer " << it->second << endl;
            abort();
        }

        string output;
        auto tp = top_layers.first;
        if (tp != top_layers.second) {
            output = tp->second;
        }

        NetworkNode params;
        if(caffe_layer.childs.size()) {
            params = caffe_layer.get_child(caffe_layer.childs.size()-1);
        }

        Layer* layer = net->add_layer(layer_type, params, vinputs, it->second, output);
        load_layer_data(layer, caffe_layer);
    }
}


unique_ptr<Net> CaffeLoader::build_network(const CaffeNetworkDescription &desc)
{
    auto net = make_unique<Net>();
    NetworkNode *root = desc.get_root();
    auto it = root->properties.find(string("input"));
    vector<string> input_layers_names;
    net->add_layer(Layer::op_input, *root, input_layers_names, it->second, it->second);

    for (auto l : root->childs) {
        if(l->node_type == NetworkNode::layer_node) {
            add_layer(net.get(), *l);
        }
    }

    cout << "\n\n\n";

    net->complete_construction();

    return net;
}


unique_ptr<Net> CaffeLoader::load_prototxt(const string &fname, const string &data_dir)
{
    data_files_path = data_dir;
    ifstream f(fname.c_str());
    stringstream ss;

    if (f.is_open()) {
        ss << f.rdbuf();
        f.close();
        cout << "opening newtwork description " << fname << '\n';

        CaffeNetworkDescription desc{ss};
        unique_ptr<Net> net = build_network(desc);

        return net;

    } else {
        cout << string("cannot open file ") + fname << endl;
        abort();
    }
}
