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

#include <turbojpeg.h>
#include <stdlib.h>
#include <assert.h>
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include "file_utils.h"
#include "load_jpeg.h"

using namespace std;

typedef unsigned char uchar;

void JpegLoader::load_and_decode_jpeg(const string &fname, RGBImage &ret)
{
    ret.w = 0;
    ret.h = 0;

    if(file_exists(fname)) {

        size_t sz = file_size(fname);
        FILE *f = fopen(fname.c_str(), "r");
        uchar *buf = (uchar*) malloc(sz);
        assert(buf);
        size_t read_sz = fread(buf, 1, sz, f);
        if(sz!=read_sz){
            cout << "error reading " << fname << endl;
            abort();
        }
        fclose(f);

        int width;
        int height;
        int jpegSubsamp;

        tjDecompressHeader2(decompressor, buf, sz, &width, &height, &jpegSubsamp);
        ret.data.resize(width*height*3);
        tjDecompress2(decompressor, buf, sz, (uchar*)ret.data.data(), width, 0/*pitch*/, height, TJPF_RGB, TJFLAG_FASTDCT);

        free(buf);

        ret.w = width;
        ret.h = height;

    } else {
        cout << "error opening file " << fname << endl;
        abort();

    }
}



