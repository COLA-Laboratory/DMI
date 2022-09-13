#include "IDTLZ.h"

double* libIDTLZ(int n_sample,int n_var, int n_obj, int testcase, double* data_in, double* data_out)
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
            idtlz1(ind_x,ind_y);
        }
        if (testcase==2){
            idtlz2(ind_x,ind_y);
        }
        if (testcase==3){
            idtlz3(ind_x,ind_y);
        }
        if (testcase==4){
            idtlz4(ind_x,ind_y);
        }

        for (j=0;j<n_obj;j++){
            data_out[i*n_obj+j] = ind_y[j];
        }
    }
    free(ind_x);
    free(ind_y);
    return 0;
}

void IDTLZ_init()
{
    return;
}

void IDTLZ_free()
{
    return;
}
