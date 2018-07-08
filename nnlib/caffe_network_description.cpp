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

#include <list>
#include <set>
#include <stack>
#include <vector>
#include <string>
#include <iostream>

#include "caffe_network_description.h"
#include "string_utils.h"

using namespace std;


CaffeNetworkDescription::CaffeNetworkDescription()
{
    root = new NetworkNode;
    init_open_of_close();
}


CaffeNetworkDescription::CaffeNetworkDescription(stringstream &ss)
{
    root = new NetworkNode;
    init_open_of_close();

    build_from_caffe_prototxt(ss);
}


CaffeNetworkDescription::~CaffeNetworkDescription()
{
    if(!root) return;

    NetworkNode *node = root;
    int ndel = 0;
    list<NetworkNode*> queue;
    set<NetworkNode*> deleted;
    queue.push_back(node);

    while(!queue.empty()) {

        NetworkNode *nn = queue.back();

        bool can_delete = true;
        for(NetworkNode *child: nn->childs) {
            if(deleted.find(child) == deleted.end()) {
                queue.push_back(child);
                can_delete = false;
            }
        }

        if(nn->childs.size() == 0 || can_delete) {
            delete nn;
            queue.remove(nn);
            deleted.insert(nn);
            ndel++;
        }

    }

}


void CaffeNetworkDescription::init_open_of_close()
{
    open_of_close.insert({'}', '{'});
    open_of_close.insert({')', '('});
    open_of_close.insert({']', '['});
}


void CaffeNetworkDescription::check_match(char ch)
{
    char top = par_stack.top();
    if(open_of_close[ch] != top) {
        cout << "parse error ! " << top  << " " << open_of_close[top] << endl;
        abort();
    }
}


void CaffeNetworkDescription::build_from_caffe_prototxt(stringstream &ss)
{
    stack<NetworkNode*> node_stack;

    NetworkNode *node = root;
    node->node_type = root_node;
    node_stack.push(node);

    string prevtok;
    while (!ss.eof()) {

        string line;
        getline(ss, line);
        vector<string> tokens = tokenize(line);

        for (auto &tok : tokens) {

            if(prevtok.find(':') != string::npos) {
                node->properties.insert({remove_char(prevtok, ':'), remove_char(tok, '"')} );
            }
            if(tok.find('{') != string::npos)
            {
                par_stack.push('{');
                node = new NetworkNode;

                if(prevtok.find("layer") != string::npos) {
                    node->node_type = layer_node;
                } else if(prevtok.find("param") != string::npos) {
                    node->node_type = layer_properties_node;
                }

                node_stack.top()->childs.push_back(node);
                node_stack.push(node);
            }

            if(tok.find('}') != string::npos)
            {
                node_stack.pop();
                check_match('}');
                par_stack.pop();
                node = node_stack.top();
            }

            prevtok = tok;

        }

    }

}


void CaffeNetworkDescription::rewrite_network(const NetworkNode *node)
{
    for(auto &prop : node->properties ) {
        cout << prop.first << ": " << prop.second << endl;
    }

    for(auto pc : node->childs) {
        if(pc->node_type == layer_node) {
            cout << "layer " << endl;
        } else if (pc->node_type == layer_properties_node) {
            cout << "properties " << endl;
        }
        cout << "{" << endl;
        rewrite_network(pc);
        cout << "}" << endl;
    }
}


void CaffeNetworkDescription::rewrite_network()
{
    rewrite_network(root);
}
