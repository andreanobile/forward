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

#include "ndarray.h"
#include <cstdio>
#include <string>
#include <memory>
#include <fstream>

using namespace std;

void ndarray::dump(const std::string &fname)
{
    ofstream file(fname.c_str(), ios::out|ios::binary|ios::trunc);

    if(file.is_open()) {
        file.write(reinterpret_cast<char*>(data), sizeof(float)*num_elements);
        file.close();
    } else {
        cout << "error opening dump file " << fname << '\n';
        abort();
    }
}


unique_ptr<ndarray> ndarray_from_file(const string &fname)
{
    streampos size;
    ifstream file (fname.c_str(), ios::in|ios::binary|ios::ate);

    if (file.is_open()) {
        size = file.tellg();
        file.seekg(0, ios::beg);
        auto data = make_unique<ndarray>();
        data->allocate({size/sizeof(float)});
        file.read(reinterpret_cast<char*>(data->get_data()), size);
        if(!file) {
            cout << "error loading file " << fname << '\n';
            abort();
        }
        file.close();
        return data;
    } else {
        cout << "error opening file " << fname << '\n';
        abort();
        return nullptr;
    }
}
