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

void zdt1 (double *xreal, double*obj);
void zdt2 (double *xreal, double*obj);
void zdt3 (double *xreal, double*obj);
void zdt4 (double *xreal, double*obj);
void zdt6 (double *xreal, double*obj);
void zdt31 (double *xreal, double*obj);
void zdt32 (double *xreal, double*obj);
void zdt33 (double *xreal, double*obj);
void zdt34 (double *xreal, double*obj);

void ZDT_init();
void ZDT_free();
double* libZDT(int n_sample,int n_var, int n_obj, int testcase, double* data_in, double* data_out);
#endif
