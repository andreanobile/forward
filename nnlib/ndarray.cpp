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
//#include "file_utils.h"

using namespace std;

void ndarray::dump(const std::string &fname)
{
    FILE *f;
    f = fopen(fname.c_str(), "w");
    fwrite(data, sizeof(float), num_elements, f);
    fclose(f);
}


unique_ptr<ndarray> ndarray_from_file(const string &fname)
{

    streampos size;
    ifstream file ("example.bin", ios::in|ios::binary|ios::ate);
    if (file.is_open()) {
        size = file.tellg();
        auto data = make_unique<ndarray>();
        data->allocate({size/sizeof(float)});
        file.read(reinterpret_cast<char*>(data->get_data()), size);
        if(!file) {
            cout << "error loading file " << fname << endl;
            abort();
        }
        file.close();
        return data;
    } else {
        cout << "error opening file " << fname << endl;
        abort();
        return nullptr;
    }
    /*
    if(file_exists(fname)) {

        size_t sz = file_size(fname);
        auto data = make_unique<ndarray>();
        vector<size_t> vsize;
        vsize.push_back(sz/sizeof(float));
        data->allocate(vsize);
        FILE *f = fopen(fname.c_str(), "r");
        size_t read_sz = fread(data->get_data(), sizeof(float), vsize[0], f);
        if(read_sz != sz/sizeof(float)) {
            cout << "error loading file " << fname << endl;
            abort();
        }
        fclose(f);
        return data;

    } else {
        cout << "error opening file " << fname << endl;
        abort();
        return nullptr;
    }
    */
}
