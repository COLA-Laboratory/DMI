# include "WFG.h"

void wfg21 (double *xreal, double*obj)
{
    int size;
    size  = number_variable;
    size  = WFG_normalise(xreal,size,wfg_temp);
    size  = WFG1_t1 (wfg_temp, size, wfg_K, wfg_temp);
    size  = WFG2_t2 (wfg_temp, size, wfg_K, wfg_temp);
    size  = WFG2_t3 (wfg_temp, size, wfg_K, number_objective, wfg_temp);
    WFG21_shape (wfg_temp, size, obj);
}

void wfg22 (double *xreal, double*obj)
{
    int size;
    size  = number_variable;
    size  = WFG_normalise(xreal,size,wfg_temp);
    size  = WFG1_t1 (wfg_temp, size, wfg_K, wfg_temp);
    size  = WFG2_t2 (wfg_temp, size, wfg_K, wfg_temp);
    size  = WFG2_t3 (wfg_temp, size, wfg_K, number_objective, wfg_temp);
    WFG22_shape (wfg_temp, size, obj);
}

void wfg23 (double *xreal, double*obj)
{
    int size;

    size  = number_variable;
    size  = WFG_normalise(xreal,size,wfg_temp);
    size  = WFG1_t1 (wfg_temp, size, wfg_K, wfg_temp);
    size  = WFG2_t2 (wfg_temp, size, wfg_K, wfg_temp);
    size  = WFG2_t3 (wfg_temp, size, wfg_K, number_objective, wfg_temp);
    WFG23_shape (wfg_temp, size, obj);

}

void wfg24 (double *xreal, double*obj)
{
    int size;

    size  = number_variable;
    size  = WFG_normalise(xreal,size,wfg_temp);
    size  = WFG1_t1 (wfg_temp, size, wfg_K, wfg_temp);
    size  = WFG2_t2 (wfg_temp, size, wfg_K, wfg_temp);
    size  = WFG2_t3 (wfg_temp, size, wfg_K, number_objective, wfg_temp);
    WFG24_shape (wfg_temp, size, obj);

}