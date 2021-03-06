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
#include <memory>
#include <algorithm>
#include <assert.h>
#include "net.h"
#include "layer_factory.h"
#include "chronometer.h"

using namespace std;


struct NetMem
{
    struct MemBlk
    {
        size_t start;
        size_t sz;
        bool in_use;

        MemBlk(size_t start, size_t sz) : start(start), sz(sz), in_use(true)
        { }
    };

    list<MemBlk> blks;
    size_t tot_size;

    NetMem()
    {
        tot_size = 0;
    }

    size_t get_blk(size_t req_amount)
    {
        //align to 64bytes
        size_t amount = req_amount;
        if(req_amount%16 != 0) {
            amount = ((req_amount/16)+1)*16;
        }

        auto bestblk = blks.end();
        size_t minsize = numeric_limits<size_t>::max();

        for (auto it = blks.begin(); it != blks.end(); ++it) {
            if(!it->in_use && it->sz >= amount) {
                if(it->sz <= minsize) {
                    minsize = min(it->sz, minsize);
                    bestblk = it;
                }
            }
        }

        if(bestblk != blks.end()) {
            if(bestblk->sz >= amount) {
                bestblk->in_use = true;

                return bestblk->start;
            }
        }

        size_t index = 0;
        size_t sz = 0;

        if(!blks.empty()) {
            auto &lstblk = blks.back();
            index = lstblk.start;
            sz = lstblk.sz;

            if(!lstblk.in_use) {
                tot_size -= lstblk.sz;
                lstblk.sz = amount;
                tot_size += amount;
                lstblk.in_use = true;
                return lstblk.start;
            }
        }

        MemBlk newblk(index + sz, amount);
        blks.push_back(newblk);
        tot_size += amount;
        return newblk.start;
    }

    void release_blk(size_t index)
    {
        for(auto &blk : blks) {
            if(index == blk.start) {
                blk.in_use = false;
                return;
            }
        }
        cout << "relase a non existsing block! " << index << endl;
        abort();
    }

};


static bool can_execute(Layer *layer, const set<string> &buffer_valid)
{
    for(Layer* l : layer->input_layers) {
        if( buffer_valid.find(l->output_name) == buffer_valid.end() ) {
            return false;
        }
    }
    return true;
}


static size_t vprod(const vector<size_t> &v) {

    if(v.empty()) {
        return 0;
    }

    size_t sz = 1;

    for(auto i: v) {
        sz *= i;
    }
    return sz;
}


bool Net::bind(const vector<size_t> &shape)
{
    InputLayer *inp;
    inp = dynamic_cast<InputLayer*>(*input_layers.begin());

    if(!shape.empty()) {
        inp->input_dims = shape;
    }

    NetMem fmap_pool;
    map<string, size_t> outname_to_index;
    map<size_t, int> idx_refcount;
    size_t nfloats = 0;
    size_t maxmem = 0;

    NetMem conv_temp_pool;
    map<Layer*, size_t> idx_padded_input;

    NetMem conv_im2col_pool;
    map<Layer*, size_t> idx_im2col;

    for(Layer* layer: vsched) {

        layer->bind();

        if(layer->op_type == Layer::op_convolution) {
            auto conv = dynamic_cast<ConvLayer*>(layer);
            if(conv->padded_input_nelem != 0u) {
                size_t idx = conv_temp_pool.get_blk(conv->padded_input_nelem);
                idx_padded_input[layer] = idx;
                conv_temp_pool.release_blk(idx);
            }
            if(!conv->im2col_buffer_shape.empty()) {
                size_t idx = conv_im2col_pool.get_blk(vprod(conv->im2col_buffer_shape));
                idx_im2col[layer] = idx;
                conv_im2col_pool.release_blk(idx);
            }
        }

        bool do_inplace = layer->inplace_possible;
        //bool do_inplace = false;

        size_t idx;

        //get idx
        if(do_inplace) {
            idx = outname_to_index[(*layer->input_layers.begin())->output_name];
            outname_to_index[layer->output_name] = idx;
        } else {
            size_t req = vprod(layer->output_shape);
            nfloats += req;
            idx = fmap_pool.get_blk(req);
            outname_to_index[layer->output_name] = idx;
        }

        //acquire
        if(layer->output_layers.empty()) {
            idx_refcount[idx]++;
        }
        idx_refcount[idx]+=layer->output_layers.size();

        maxmem = max(nfloats, maxmem);

        //release
        for(Layer* li: layer->input_layers) {

            size_t li_idx = outname_to_index[li->output_name];
            idx_refcount[li_idx]--;

            if(idx_refcount[li_idx] == 0) {
                fmap_pool.release_blk(li_idx);
                nfloats -= vprod(li->output_shape);
                maxmem = max(nfloats, maxmem);
            }
        }

    }

    cout << "memory pool for feature maps -> blocks ref count at end:" << endl;
    for(auto &p : idx_refcount) {
        cout << "block_offset: " << p.first << " ref_count:" << p.second << endl;
    }

    cout << "maxmem = " << maxmem << " floats" << endl;
    cout << "pool maxmem = " << fmap_pool.tot_size << " floats" << endl;

    fmap_buffer.allocate(fmap_pool.tot_size);
    conv_pad_buffer.allocate(conv_temp_pool.tot_size);
    conv_im2col_buffer.allocate(conv_im2col_pool.tot_size);

    for(auto layer : layers) {
        shared_ptr<ndarray> oarr(new ndarray());
        oarr->attach(&fmap_buffer[outname_to_index[layer->output_name]], layer->output_shape);
        layer->output_array = oarr;
        layer->input_arrays.clear();
        for(Layer *inl : layer->input_layers) {
            shared_ptr<ndarray> arr(new ndarray());
            arr->attach(&fmap_buffer[outname_to_index[inl->output_name]], inl->output_shape);
            layer->input_arrays.push_back(arr);
        }

        if(layer->op_type == Layer::op_convolution) {
            auto conv = dynamic_cast<ConvLayer*>(layer.get());
            if(conv->padded_input_nelem != 0u) {
                conv->padded_input = make_shared<ndarray>();
                conv->padded_input->attach(&conv_pad_buffer[idx_padded_input[layer.get()]], {conv->padded_input_nelem});
            }
            if(!conv->im2col_buffer_shape.empty()) {
                conv->im2col_buffer = make_shared<ndarray>();
                conv->im2col_buffer->attach(&conv_im2col_buffer[idx_im2col[layer.get()]], conv->im2col_buffer_shape);
            }
        }
    }

    return true;
}


void Net::schedule()
{
    vsched.clear();
    list<Layer*> pqueue;
    set<string> buffer_valid;
    Layer *lay;

    lay = *input_layers.begin();
    pqueue.push_back(lay);
    auto it = pqueue.begin();
    bool rstart = false;
    bool no_exec = true;

    while(!pqueue.empty()) {

        lay = *it;

        if(can_execute(lay, buffer_valid)) {
            buffer_valid.insert(lay->output_name);
            vsched.push_back(lay);
            cout << "pushed layer " << lay->name << " to execution" << endl;
            pqueue.remove(lay);
            for(auto l : lay->output_layers) {
                //add layer to processing queue if not already added
                if(find(pqueue.begin(), pqueue.end(), l) == pqueue.end()) {
                    pqueue.push_front(l); //FRONT = DFS, BACK = BFS
                }
            }
            it = pqueue.begin();
            no_exec = false;
        } else {
            it++;
            if(it == pqueue.end()) {
                it = pqueue.begin();
                if(no_exec && rstart) {
                    cout << "no layer can be executed !!!" << endl;
                    for(auto l: pqueue) {
                        cout << l->name << endl;
                    }
                    abort();
                }
                rstart = true;
                no_exec = true;
            }
        }
    }

    assert(vsched.size() == layers.size());
    cout << "layers count = " << layers.size() << endl;
}


void Net::optimize()
{
    int relu_count = 0;
    int removed_relu_count = 0;
    int scale_count = 0;
    int removed_scale_count = 0;

    vector<shared_ptr<Layer>> vl;

    for(auto &layer : layers) {
        vl.push_back(layer);
    }

    for(auto &layer : vl) {

        auto it = layer->input_layers.begin();
        if(it != layer->input_layers.end()) {
            Layer* inp_layer = *it;

            if(layer->op_type == Layer::op_relu) {
                relu_count++;
                if(inp_layer->can_do_relu) {
                    inp_layer->relu = true;
                    remove_layer(layer);
                    removed_relu_count++;
                    //cout << "merged " << layer->name << " with " << inp_layer->name << endl;
                }
            }

            else if(layer->op_type == Layer::op_scale) {
                scale_count++;
                if(inp_layer->op_type == Layer::op_batchnorm) {
                    //merge batchnorm with scale (if bn has not been merged with relu)
                    auto bn = dynamic_cast<BatchNormLayer*>(inp_layer);
                    if(!bn->relu) {
                        bn->relu = layer->relu;
                        bn->set_scale_layer(static_pointer_cast<ScaleLayer>(layer));

                        remove_layer(layer);
                        removed_scale_count++;
                    }
                }
            }
        }
    }

    cout << "removed " << removed_relu_count << " relu layers out of " << relu_count << endl;
    cout << "removed " << removed_scale_count << " scale layers out of " << scale_count << endl;
}


void Net::add_layer(const shared_ptr<Layer> &layer)
{
    layers.push_back(layer);
    pair<string, Layer*> po(layer->output_name, layer.get());
    out_name_to_layer.insert(po);
}


void Net::remove_layer(const shared_ptr<Layer> &layer)
{
    //assign to all input layers, the current layer's output layers as output layers
    for(Layer *il : layer->input_layers) {
        il->output_layers.remove(layer.get());
        for(Layer *ol : layer->output_layers) {
            il->output_layers.push_back(ol);
        }
    }

    //assign to all output layers, the current layer's input layers as input layers
    for(Layer *ol : layer->output_layers) {
        ol->input_layers.remove(layer.get());
        for(Layer *il : layer->input_layers) {
            ol->input_layers.push_back(il);
        }
    }

    out_name_to_layer.erase(layer->output_name);
    layers.remove(layer);
}


Layer* Net::add_layer(Layer::Type layer_type, const NetworkNode &params,
                                 const vector<string> &inputs, const string &layer_name,
                                 const string &output_name)
{
    shared_ptr<Layer> layer = create_layer(layer_type, params);
    layer->name = layer_name;

    for(auto &input_name : inputs ) {
        string str = rename_table.get_rename(input_name);
        Layer *input_to_layer = output_name_to_layer(str);
        layer->add_input(input_to_layer);
        input_to_layer->add_output(layer.get());
    }

    string out_name;

    if(output_name.length() > 0) {
        out_name = output_name;
    } else {
        out_name = layer_name;
    }

    string str = rename_table.create_rename(out_name);
    layer->output_name = str;

    add_layer(layer);

    return layer.get();
}


void Net::complete_construction()
{
    output_layers.clear();
    input_layers.clear();

    for (auto &layer : layers) {

        if(layer->output_layers.empty()) {
            output_layers.push_back(layer.get());
            cout << "output layer: " << layer->name << " output name: " << layer->output_name << endl;
        }

        if(layer->input_layers.empty()) {
            input_layers.push_back(layer.get());
            cout << "input layer: " << layer->name << " output name: " << layer->output_name << endl;
        }

    }

    rename_table.clear();

    optimize();
    schedule();
}


void Net::forward()
{
    Chronometer time;

    time.start();
    for(Layer* layer : vsched) {
        layer->forward();
    }
    cout << "forward time = " << time.stop() << " s" << endl;
}


void Net::copy_net_sharing_weights(const Net &original)
{
    //need to create layers in this net, copying the layers of the original network
    map<Layer*, Layer*> old_to_new;

    for(auto &ol : original.layers) {
        shared_ptr<Layer> nl = copy_layer(ol.get());
        old_to_new[ol.get()] = nl.get();
        layers.push_back(nl);
    }

    //for each new layer, set input and output layers
    for(auto &ol : original.layers) {

        auto nl = old_to_new[ol.get()];
        nl->input_layers.clear();
        for(auto io : ol->input_layers) {
            nl->input_layers.push_back(old_to_new[io]);
        }
        nl->output_layers.clear();
        for(auto oo : ol->output_layers) {
            nl->output_layers.push_back(old_to_new[oo]);
        }
    }

    input_layers.clear();
    for(auto &ol : original.input_layers) {
        input_layers.push_back(old_to_new[ol]);
    }

    output_layers.clear();
    for(auto &ol : original.output_layers) {
        output_layers.push_back(old_to_new[ol]);
    }

    out_name_to_layer.clear();
    for(auto &on : original.out_name_to_layer) {
        out_name_to_layer[on.first] = old_to_new[on.second];
    }

    vsched.clear();
    for(auto &l: original.vsched) {
        vsched.push_back(old_to_new[l]);
    }
}
