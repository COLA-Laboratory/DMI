/*
 * minusDTLZ1.c
 *
 * Authors:
 *  Minhui Liao <minhui.liao1@gmail.com>
 *  Ke Li <k.li@exeter.ac.uk>
 *
 * Institution:
 *  COLA-Laboratory @ University of Exeter | http://cola-laboratory.github.io
 *
 * Copyright (c) 2020 Minhui Liao, Ke Li
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

#include"IDTLZ.h"

void idtlz1 (double *xreal, double*obj)
{
    int i, j, k;
    int aux;
    double gx;

    gx = 0.0;
    k  = number_variable - number_objective + 1;
    for(i = number_variable - k; i < number_variable; i++)
        gx += pow((xreal[i] - 0.5), 2.0) - cos(20.0 * PI * (xreal[i] - 0.5));
    gx = 100.0 * (k + gx);

    for (i = 0; i < number_objective; i++)
        obj[i] = (1.0 + gx) * 0.5;

    for (i = 0; i < number_objective; i++)
    {
        for (j = 0; j < number_objective - (i + 1); j++)
            obj[i] *= xreal[j];
        if (i != 0)
        {
            aux     = number_objective - (i + 1);
            obj[i] *= 1 - xreal[aux];
        }
        obj[i] = 0.5 * (1+gx) - obj[i];
    }

    return;
}
