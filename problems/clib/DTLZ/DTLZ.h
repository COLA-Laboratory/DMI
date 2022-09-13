#ifndef SAMARITAN_MDTLZ_H
#define SAMARITAN_MDTLZ_H
# include <pthread.h>
# include <stdio.h>
# include <stdlib.h>
# include <stdarg.h>
# include <time.h>
# include <math.h>
# include <float.h>
# include <string.h>
# include <unistd.h>
# include <sys/types.h>
# include <sys/stat.h>

# define PI  M_PI

double *theta;

int number_variable;
int number_objective;

void dtlz1 (double *xreal, double*obj);
void dtlz2 (double *xreal, double*obj);
void dtlz3 (double *xreal, double*obj);
void dtlz4 (double *xreal, double*obj);
void dtlz5 (double *xreal, double*obj);
void dtlz6 (double *xreal, double*obj);
void dtlz7 (double *xreal, double*obj);
void dtlz71 (double *xreal, double*obj);
void dtlz72 (double *xreal, double*obj);
void dtlz73 (double *xreal, double*obj);

void DTLZ_init();
void DTLZ_free();

double* libDTLZ(int n_sample,int n_var, int n_obj, int testcase, double* data_in, double* data_out);
#endif