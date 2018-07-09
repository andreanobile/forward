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

#ifndef CAFFENETWORKDESCRIPTION_H
#define CAFFENETWORKDESCRIPTION_H

#include <sstream>
#include "network_node.h"

class CaffeNetworkDescription
{
    std::map<char, char> open_of_close;
    std::stack<char> par_stack;

    void check_match(char ch);
    void init_open_of_close();
    void rewrite_network(const NetworkNode *node);

    NetworkNode *root;

public:

    CaffeNetworkDescription();
    CaffeNetworkDescription(std::stringstream &ss);
    ~CaffeNetworkDescription();

    NetworkNode *get_root() const
    {
        return root;
    }

    void build_from_caffe_prototxt(std::stringstream &ss);
    void rewrite_network();

};

#endif // CAFFENETWORKDESCRIPTION_H
