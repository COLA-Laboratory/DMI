/*
 * WFG2.c
 *
 * Authors:
 *  Ke Li <k.li@exeter.ac.uk>
 *  Renzhi Chen <rxc332@cs.bham.ac.uk>
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

# include "WFG.h"

void wfg2 (double *xreal, double*obj)
{
    int size;
    size  = number_variable;
    size  = WFG_normalise(xreal,size,wfg_temp);
    size  = WFG1_t1 (wfg_temp, size, wfg_K, wfg_temp);
    size  = WFG2_t2 (wfg_temp, size, wfg_K, wfg_temp);
    size  = WFG2_t3 (wfg_temp, size, wfg_K, number_objective, wfg_temp);
    WFG2_shape (wfg_temp, size, obj);
}