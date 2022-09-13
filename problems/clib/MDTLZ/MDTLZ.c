#include "MDTLZ.h"

double* libMDTLZ(int n_sample,int n_var, int n_obj, int testcase, double* data_in, double* data_out)
{

    int i,j;
    // setting global parameters
    number_variable = n_var;
    number_objective = n_obj;

    // malloc memory
    double* ind_x = malloc(n_var * sizeof(double));
    double* ind_y = malloc(n_obj * sizeof(double));

    for (i=0; i<n_sample; i++){
        for (j=0;j<n_var;j++){
            ind_x[j] = data_in[i*n_var+j];
        }
        if (testcase==1){
            mdtlz1(ind_x,ind_y);
        }
        if (testcase==2){
            mdtlz2(ind_x,ind_y);
        }
        if (testcase==3){
            mdtlz3(ind_x,ind_y);
        }
        if (testcase==4){
            mdtlz4(ind_x,ind_y);
        }


        for (j=0;j<n_obj;j++){
            data_out[i*n_obj+j] = ind_y[j] ;
        }
    }
    free(ind_x);
    free(ind_y);
    return 0;
}


void MDTLZ_init()
{
    gx    = malloc (sizeof(double) * number_objective);
    h     = malloc (sizeof(double) * number_objective);
    temp_xreal = malloc (sizeof(double) * number_variable);
    return;
}

void MDTLZ_free()
{
    free (gx);
    free (h);
    free(temp_xreal);
    return;
}
