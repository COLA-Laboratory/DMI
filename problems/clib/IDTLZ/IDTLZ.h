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

double *gx;
double *h;
double *temp_xreal;

int number_variable;
int number_objective;

void idtlz1 (double *xreal, double*obj);
void idtlz2 (double *xreal, double*obj);
void idtlz3 (double *xreal, double*obj);
void idtlz4 (double *xreal, double*obj);

void IDTLZ_init();
void IDTLZ_free();

double* libIDTLZ(int n_sample,int n_var, int n_obj, int testcase, double* data_in, double* data_out);
#endif
