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

#include <string>
#include <vector>
#include <algorithm>
#include "string_utils.h"

using namespace std;


vector<string> split_string(string str, string sep)
{
    size_t start, stop;
    string piece;
    vector<string> vstring;
    size_t len = str.length();

    start = 0;
    do {
        stop = str.find(sep, start);
        piece = str.substr(start, stop-start);
        if(stop-start > 0){
            vstring.push_back(piece);
        }

        start = stop+1;

    } while (stop != str.npos && start < len);

    return vstring;
}


string remove_char(string &str, char ch)
{
    string output;
    size_t len = str.length();
    output.reserve(len);

    for (size_t i=0;i<len;i++) {
        if(str[i] != ch)
        output += str[i];
    }
    return output;
}


vector<string> tokenize(string &str)
{
    vector<string> tokens;
    vector<string> lines = split_string(str, "\n");

    for (auto &line : lines) {
        if(line.length() > 0) {
            replace(line.begin(), line.end(), '\t', ' ');
            vector<string> vstr = split_string(line, " ");
            //cout << line << endl;
            for (auto &piece : vstr) {
                //cout << piece << endl;
                tokens.push_back(piece);
            }
        }
    }
    return tokens;
}
