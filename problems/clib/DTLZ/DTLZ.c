#include "DTLZ.h"

double* libDTLZ(int n_sample,int n_var, int n_obj, int testcase, double* data_in, double* data_out)
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
            dtlz1(ind_x,ind_y);
        }
        if (testcase==2){
            dtlz2(ind_x,ind_y);
        }
        if (testcase==3){
            dtlz3(ind_x,ind_y);
        }
        if (testcase==4){
            dtlz4(ind_x,ind_y);
        }
        if (testcase==5){
            dtlz5(ind_x,ind_y);
        }
        if (testcase==6){
            dtlz6(ind_x,ind_y);
        }
        if (testcase==7){
            dtlz7(ind_x,ind_y);
        }
        if (testcase==71){
            dtlz71(ind_x,ind_y);
        }
        if (testcase==72){
            dtlz72(ind_x,ind_y);
        }
        if (testcase==73){
            dtlz73(ind_x,ind_y);
        }
        for (j=0;j<n_obj;j++){
            data_out[i*n_obj+j] = ind_y[j] ;
        }
    }
    free(ind_x);
    free(ind_y);
    return 0;
}

void DTLZ_init()
{
    theta = (double *) malloc (number_variable * sizeof(double));
    return;
}

void DTLZ_free()
{
    free(theta);
    return;
}