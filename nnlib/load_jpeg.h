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

#ifndef LOAD_JPEG_H
#define LOAD_JPEG_H

#include <vector>
#include <string>
#include <memory>
#include <turbojpeg.h>

struct RGBImage
{
    std::vector<unsigned char> data;
    int w;
    int h;
};


class JpegLoader
{
    tjhandle decompressor;
public:
    JpegLoader()
    {
        decompressor = tjInitDecompress();
    }
    ~JpegLoader()
    {
        tjDestroy(decompressor);
    }

    void load_and_decode_jpeg(const std::string &fname, RGBImage &ret);

};


#endif // LOAD_JPEG_H
