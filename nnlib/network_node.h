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

#ifndef NETWORK_NODE_H
#define NETWORK_NODE_H

#include <map>
#include <vector>
#include <string>
#include <memory>
#include <utility>

enum NodeType {root_node, layer_node, layer_properties_node};


struct NetworkNode
{
    NodeType node_type;
    std::multimap<std::string, std::string> properties;
    std::vector<NetworkNode*> childs;

    std::string get_inner_prop(const std::string &prop) const
    {
        for(auto &p : childs) {
            auto it = p->properties.find(prop);
            if(it != p->properties.end()) {
                return it->second;
            }
        }

        std::string ret;
        return ret;
    }

    size_t get_int_inner_prop(const std::string &prop) const
    {
        return std::stoi(get_inner_prop(prop));
    }

    bool get_bool_inner_prop(const std::string &prop) const
    {
        std::string str = get_inner_prop(prop);
        bool ret;
        if(str.length() == 0) {
            ret = false;
        } else {
            if(str.compare("false") == 0) {
                ret = false;
            }
            else {
                ret = true;
            }
        }
        return ret;
    }

    std::vector<std::string> get_prop(const std::string &prop) const
    {
        std::vector<std::string> vprop;
        auto rg = properties.equal_range(prop);
        for(auto it=rg.first; it != rg.second; it++) {
            vprop.push_back(it->second);
        }
        return vprop;
    }

    void set_prop(const std::string &key, const std::string &value)
    {
        properties.insert(std::make_pair(key, value));
    }

    void set_prop(const std::string &key, size_t value)
    {
        set_prop(key, std::to_string(value));
    }

    NetworkNode &get_child(size_t idx) const
    {
        return *(childs[idx]);
    }

    void clear()
    {
        childs.clear();
        properties.clear();
    }

};

#endif // NETWORK_NODE_H
