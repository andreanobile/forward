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

#ifndef CHRONO_H
#define CHRONO_H

#include <sys/time.h>
class Chrono
{
    struct timeval tv;
public:
    Chrono() {
        //gettimeofday(&tv, nullptr);
    }

    void start()
    {
        gettimeofday(&tv, nullptr);
    }

    double stop()
    {
        double time_us, nowtime_us;
        struct timeval nowtv;
        gettimeofday(&nowtv, nullptr);
        time_us = ((double)tv.tv_sec) * 1000000.0 + ((double)tv.tv_usec);
        nowtime_us = ((double)nowtv.tv_sec) * 1000000.0 + ((double)nowtv.tv_usec);
        return (nowtime_us-time_us)/1000000.0;
    }
};

#endif // CHRONO_H
