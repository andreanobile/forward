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

#ifndef CHRONOMETER_H
#define CHRONOMETER_H

#include <ctime>
#include <ratio>
#include <chrono>

class Chronometer
{
    std::chrono::steady_clock::time_point t1;
public:
    Chronometer()
    {
    }

    void start()
    {
        using namespace std::chrono;
        t1 = steady_clock::now();
    }

    double stop()
    {
        using namespace std::chrono;
        steady_clock::time_point t2 = steady_clock::now();
        duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
        return static_cast<double>(time_span.count());
    }
};

#endif // CHRONOMETER_H
