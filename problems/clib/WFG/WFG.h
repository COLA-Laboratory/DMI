#ifndef SAMARITAN_TOOLKIT_H
#define SAMARITAN_TOOLKIT_H

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

double *wfg_temp;
double *temp;
double *wfg_w;

int number_variable;
int number_objective;
int wfg_K;

void WFG_ini ();
void WFG_free ();
int WFG_normalise (double *z, int z_size, double *result);

int WFG1_t1 (double *y, int y_size, int k, double *result);
int WFG1_t2 (double *y, int y_size, int k, double *result);
int WFG1_t3 (double *y, int y_size, double *result);
int WFG1_t4 (double *y, int y_size, int k, int M, double *result);
int WFG2_t2 (double *y, int y_size, int k, double *result);
int WFG2_t3 (double *y, int y_size, int k, int M, double *result);
int WFG4_t1 (double *y, int y_size, double *result);
int WFG5_t1 (double *y, int y_size, double *result);
int WFG6_t2 (double *y, int y_size, int k, const int M, double *result);
int WFG7_t1 (double *y, int y_size, int k, double *result);
int WFG8_t1 (double *y, int y_size, int k, double *result);
int WFG9_t1 (double *y, int y_size, double *result);
int WFG9_t2 (double *y, int y_size, int k, double * result);

void WFG1_shape (double *y, int size, double *result);
void WFG2_shape (double *y, int size, double *result);
void WFG3_shape (double *y, int size, double *result);
void WFG4_shape (double *y, int size, double *result);
void WFG42_shape (double *y, int size, double *result);
void WFG43_shape (double *y, int size, double *result);
void WFG44_shape (double *y, int size, double *result);
void WFG45_shape (double *y, int size, double *result);
void WFG46_shape (double *y, int size, double *result);
void WFG47_shape (double *y, int size, double *result);
void WFG48_shape (double *y, int size, double *result);
void WFG21_shape (double *y, int size, double *result);
void WFG22_shape (double *y, int size, double *result);
void WFG23_shape (double *y, int size, double *result);
void WFG24_shape (double *y, int size, double *result);

void wfg1 (double *xreal, double*obj);
void wfg2 (double *xreal, double*obj);
void wfg3 (double *xreal, double*obj);
void wfg4 (double *xreal, double*obj);
void wfg5 (double *xreal, double*obj);
void wfg6 (double *xreal, double*obj);
void wfg7 (double *xreal, double*obj);
void wfg8 (double *xreal, double*obj);
void wfg9 (double *xreal, double*obj);
void wfg42 (double *xreal, double*obj);
void wfg43 (double *xreal, double*obj);
void wfg44 (double *xreal, double*obj);
void wfg45 (double *xreal, double*obj);
void wfg46 (double *xreal, double*obj);
void wfg47 (double *xreal, double*obj);
void wfg48 (double *xreal, double*obj);
void wfg21 (double *xreal, double*obj);
void wfg22 (double *xreal, double*obj);
void wfg23 (double *xreal, double*obj);
void wfg24 (double *xreal, double*obj);

int libWFG(int n_sample,int n_var, int n_obj, int testcase, double* data_in, double* data_out);
int libWFGShape(int n_sample, int size, int testcase, double* data_in, double* data_out);
#endif //SAMARITAN_TOOLKIT_H
