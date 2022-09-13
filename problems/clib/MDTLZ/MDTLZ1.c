/*
 * mDTLZ1.c
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
#include "MDTLZ.h"

void mdtlz1 (double *xreal, double*obj)
{
    int i, j;
    int length, index;

    MDTLZ_init();

    for (i = 0; i < number_objective; i++)
    {
        gx[i]  = 0.0;
        length = (number_variable - number_objective - i ) / number_objective + 1;

        for (j = 0; j < length; j++)
        {
          index = number_objective + i + j * number_objective - 1;
          gx[i] +=  pow ((xreal[index] - 0.5), 2) - cos (20.0 * PI * (xreal[index] - 0.5));
        }
         gx[i] = 100 * (length + gx[i]);
    }

    h[0] = 1;
    for (i = 0; i < number_objective - 1; i++)
        h[0] = h[0] * xreal[i];
    h[0] = 0.5 * (1 - h[0]);

    for (i = 1; i < number_objective - 1; i++)
    {
       h[i] = 1;
       for (j = 0; j < number_objective - i - 1; j++)
           h[i] *= xreal[j];

       h[i] = 0.5 *(1 - h[i] * (1 - xreal[number_objective - 1 - i]));
    }

    h[number_objective-1] = 0.5 * xreal[0];
    for(i = 0; i < number_objective; i++)
        obj[i] = h[i] * (1 + gx[i]);
    MDTLZ_free();
    return;
}
