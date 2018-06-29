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

#include "rename_table.h"

using namespace std;

string RenameTable::create_rename(const string &str)
{
    auto r = table.insert(make_pair(str, str));
    if(r.second) {
        return str;
    } else  { //element already exist
        map<string, string>::iterator it;
        it = r.first;
        it->second += "_x";
        return it->second;
    }
}


string RenameTable::get_rename(const string &str)
{
    auto it = table.find(str);
    if (it!=table.end()) {
        return it->second;
    } else {
        std::cout << "cannot find rename! \n";
        abort();
    }
}
