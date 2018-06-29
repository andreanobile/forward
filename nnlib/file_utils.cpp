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

#include "file_utils.h"
#include <sys/stat.h>


using namespace std;

bool file_exists(const string &fname)
{
    struct stat st;
    return (stat(fname.c_str(), &st) == 0);
}


size_t file_size(const string &fname)
{
    struct stat st;
    stat(fname.c_str(), &st);
    return st.st_size;
}


string path_join(const string &first, const string &second)
{
    string ret;
    ret = first + "/" + second;
    return ret;
}
