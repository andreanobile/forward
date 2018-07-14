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

#ifndef NDARRAY_H
#define NDARRAY_H

#include <vector>
#include <malloc.h>
#include <iostream>
#include <assert.h>
#include <cstring>
#include <memory>
#include <immintrin.h>

class ndarray
{
    size_t num_elements;
    float *data;
    bool own_data;
    volatile unsigned long lock;

public:

    std::vector<size_t> shape;

    ndarray &operator=(ndarray &a) = delete;
    ndarray &operator=(const ndarray &a) = delete;

    volatile unsigned long *get_lock_address()
    {
        return &lock;
    }

    ndarray()
    {
        data = nullptr;
        lock = 0;
        num_elements = 0;
        own_data = true;
    }

    explicit ndarray(const std::vector<size_t> &s)
    {
        data = nullptr;
        lock = 0;
        allocate(s);
    }

    ~ndarray()
    {
        clear();
    }

    float *get_data()
    {
        return data;
    }

    float *get_data() const
    {
        return data;
    }

    size_t n_elements() const
    {
        return num_elements;
    }

    void allocate(const std::vector<size_t> &s)
    {
        clear();
        own_data = true;

        if(s.empty()) {return;}
        size_t sz = num_elements_from_shape(s);

        int ret = posix_memalign((void**)&data, 64, sizeof(float)*sz);
        if(ret != 0) {
            std::cout << "failed to allocate ndarray" << std::endl;
            abort();
        }
        shape = s;
        num_elements = sz;
    }

    void attach(float *buf, const std::vector<size_t> &s)
    {
        clear();
        shape = s;
        data = buf;
        own_data = false;
        num_elements = num_elements_from_shape(s);
    }

    size_t num_elements_from_shape(const std::vector<size_t> &s) const
    {
        if(s.empty()) {return 0;}

        size_t sz = 1;
        for(size_t i : s) {
            sz *= i;
        }
        return sz;
    }

    size_t num_elements_from_shape() const
    {
        return num_elements_from_shape(shape);
    }

    void clear()
    {
        shape.clear();
        num_elements = 0;
        if(data && own_data) {
            free(data);
            data = nullptr;
        }
    }

    void reshape(const std::vector<size_t> &newshape)
    {
        assert(num_elements == num_elements_from_shape(newshape));
        shape = newshape;
    }

    void zero()
    {
        memset(data, 0, num_elements*sizeof(float));
    }

    float &element(int i0, int i1)
    {
        int index = i0*shape[1] + i1;
        //assert(index < num_elements);
        return data[index];
    }

    float &element(int i0, int i1, int i2, int i3)
    {
        size_t index = i0*shape[1]*shape[2]*shape[3] + i1*shape[2]*shape[3] + i2*shape[3] + i3;
        //assert(index < num_elements);
        return data[index];
    }

    void dump(const std::string &fname);

    void copy_from(const ndarray &arr) {
        assert(num_elements == arr.n_elements());
        memcpy(data, arr.get_data(), num_elements*sizeof(float));
        reshape(arr.shape);
    }

};


std::unique_ptr<ndarray> ndarray_from_file(const std::string &fname);

#endif // NDARRAY_H
