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

#include "layer.h"
#include <cstdio>
#include <iostream>

using namespace std;

#define LOG_BIND

void Layer::log_bind()
{
#ifdef LOG_BIND
    cout << "bind " << name << " in_shape: ";
    for(auto &v : input_shapes) {
        cout << "( ";
        for(auto i: v) {
            cout << i << " ";
        }
        cout << ")  ";
    }
    cout << endl;

    cout << "bind " << name << " out_shape: ";
    cout << "( ";
    for(auto i: output_shape) {
        cout << i << " ";
    }
    cout << ")  " << endl;
#endif
}
