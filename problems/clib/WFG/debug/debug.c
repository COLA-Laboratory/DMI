#include "../WFG.h"
#include <stdio.h>


int main(){

    int n_var = 3;
    int n_obj = 2;
    int n_sample = 2;
    int testcase = 4;
    double data_in[6] = {0.0*2,0.35*4,0.35*6,1.0*2,0.35*4,0.35*6};
    double data_out[4];
    for (int i=0;i<6;i++){
        printf("%lf\t",data_in[i]);
    }
    printf("\n");
    libWFG(n_sample,n_var,n_obj,4,data_in,data_out);
    for (int i=0;i<4;i++){
        printf("%lf\t",data_out[i]);
    }
    printf("\n");
    return 0;
}
