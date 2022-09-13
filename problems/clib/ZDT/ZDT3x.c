/*
 * ZDT3.c
 *
 * Authors:
 *  Renzhi Chen <rxc332@cs.bham.ac.uk>
 *  Ke Li <k.li@exeter.ac.uk>
 *
 * Copyright (c) 2017 Renzhi Chen, Ke Li
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "ZDT.h"

void zdt31 (double *xreal, double*obj)
{
    int i;
    double f1, f2, g, h;
    double A = 20.0;
    double alpha = 0.0;
    double beta = 1.0;
    f1 = xreal[0];

    g = 0.0;
    for (i = 1; i < number_variable; i++)
        g += xreal[i];
    g = 9.0 * g / (number_variable - 1) + 1.0;
    h = 1.0 - sqrt (f1 / g) - (f1 / g) * pow(f1,alpha) * sin (A * PI * pow(f1,beta));

    f2 = g * h;

    obj[0] = f1;
    obj[1] = f2;
}


void zdt34 (double *xreal, double*obj)
{
    int i;
    double f1, f2, g, h;
    double A = 10.0;
    double alpha = 0.3;
    double beta = 1.0;
    f1 = xreal[0];

    g = 0.0;
    for (i = 1; i < number_variable; i++)
        g += xreal[i];
    g = 9.0 * g / (number_variable - 1) + 1.0;
    h = 1.0 - sqrt (f1 / g) - (f1 / g) * pow(f1,alpha) * sin (A * PI * pow(f1,beta));

    f2 = g * h;

    obj[0] = f1;
    obj[1] = f2;
}


void zdt33 (double *xreal, double*obj)
{
    int i;
    double f1, f2, g, h;
    double A = 10.0;
    double alpha = 10.0;
    double beta = 1.0;
    f1 = xreal[0];

    g = 0.0;
    for (i = 1; i < number_variable; i++)
        g += xreal[i];
    g = 9.0 * g / (number_variable - 1) + 1.0;
    h = 1.0 - sqrt (f1 / g) - (f1 / g) * pow(f1,alpha) * sin (A * PI * pow(f1,beta));

    f2 = g * h;

    obj[0] = f1;
    obj[1] = f2;
}


void zdt32 (double *xreal, double*obj)
{
    int i;
    double f1, f2, g, h;
    double A = 5.0;
    double alpha = 0.0;
    double beta = 5.0;
    f1 = xreal[0];

    g = 0.0;
    for (i = 1; i < number_variable; i++)
        g += xreal[i];
    g = 9.0 * g / (number_variable - 1) + 1.0;
    h = 1.0 - sqrt (f1 / g) - (f1 / g) * pow(f1,alpha) * sin (A * PI * pow(f1,beta));

    f2 = g * h;

    obj[0] = f1;
    obj[1] = f2;
}