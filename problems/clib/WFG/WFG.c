#include "WFG.h"

int libWFGShape(int n_sample, int size, int testcase, double* data_in, double* data_out)
{

    int i,j;

    // setting global parameters
    number_variable = size;
    number_objective = size;
    wfg_K = size - 1;
    WFG_ini ();
    // malloc memory
    double* ind_x = malloc(size * sizeof(double));
    double* ind_y = malloc(size * sizeof(double));

    /*
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
    */
    for (i=0; i<n_sample; i++){
        for (j=0;j<size;j++){
            ind_x[j] = data_in[i*size+j];
        }
        if (testcase==1){
            WFG1_shape(ind_x,size,ind_y);
        }
        if (testcase==2){
            WFG2_shape(ind_x,size,ind_y);
        }
        if (testcase==3){
            WFG3_shape(ind_x,size,ind_y);
        }
        if (testcase==4 || testcase==5 || testcase==6 || testcase==7 || testcase==8 || testcase==9){
            WFG4_shape(ind_x,size,ind_y);
        }
        if (testcase==42){
            WFG42_shape(ind_x,size,ind_y);
        }
        if (testcase==43){
            WFG43_shape(ind_x,size,ind_y);
        }
        if (testcase==44){
            WFG44_shape(ind_x,size,ind_y);
        }
        if (testcase==45){
            WFG45_shape(ind_x,size,ind_y);
        }
        if (testcase==46){
            WFG46_shape(ind_x,size,ind_y);
        }
        if (testcase==47){
            WFG47_shape(ind_x,size,ind_y);
        }
        if (testcase==48){
            WFG48_shape(ind_x,size,ind_y);
        }
        if (testcase==21){
            WFG21_shape(ind_x,size,ind_y);
        }
        if (testcase==22){
            WFG22_shape(ind_x,size,ind_y);
        }
        if (testcase==23){
            WFG23_shape(ind_x,size,ind_y);
        }
        if (testcase==24){
            WFG24_shape(ind_x,size,ind_y);
        }
        for (j=0;j<size;j++){
            data_out[i*size+j] = ind_y[j] ;
        }
    }
    WFG_free();
    free(ind_x);
    free(ind_y);
    return 0;
}

int libWFG(int n_sample,int n_var, int n_obj, int testcase, double* data_in, double* data_out)
{

    int i,j;


    // setting global parameters
    number_variable = n_var;
    number_objective = n_obj;
    wfg_K = n_obj - 1;
    WFG_ini ();
    // malloc memory
    double* ind_x = malloc(n_var * sizeof(double));
    double* ind_y = malloc(n_obj * sizeof(double));

    for (i=0; i<n_sample; i++){
        for (j=0;j<n_var;j++){
            ind_x[j] = data_in[i*n_var+j];
        }
        if (testcase==1){
            wfg1(ind_x,ind_y);
        }
        if (testcase==2){
            wfg2(ind_x,ind_y);
        }
        if (testcase==3){
            wfg3(ind_x,ind_y);
        }
        if (testcase==4){
            wfg4(ind_x,ind_y);
        }
        if (testcase==5){
            wfg5(ind_x,ind_y);
        }
        if (testcase==6){
            wfg6(ind_x,ind_y);
        }
        if (testcase==7){
            wfg7(ind_x,ind_y);
        }
        if (testcase==8){
            wfg8(ind_x,ind_y);
        }
        if (testcase==9){
            wfg9(ind_x,ind_y);
        }
        if (testcase==42){
            wfg42(ind_x,ind_y);
        }
        if (testcase==43){
            wfg43(ind_x,ind_y);
        }
        if (testcase==44){
            wfg44(ind_x,ind_y);
        }
        if (testcase==45){
            wfg45(ind_x,ind_y);
        }
        if (testcase==46){
            wfg46(ind_x,ind_y);
        }
        if (testcase==47){
            wfg47(ind_x,ind_y);
        }
        if (testcase==48){
            wfg48(ind_x,ind_y);
        }
        if (testcase==21){
            wfg21(ind_x,ind_y);
        }
        if (testcase==22){
            wfg22(ind_x,ind_y);
        }
        if (testcase==23){
            wfg23(ind_x,ind_y);
        }
        if (testcase==24){
            wfg24(ind_x,ind_y);
        }
        for (j=0;j<n_obj;j++){
            data_out[i*n_obj+j] = ind_y[j] ;
        }
    }
    WFG_free ();
    free(ind_x);
    free(ind_y);
    return 0;
}



int next_int (char *st, int st_len, int pos)
{
    int i;
    int re   = 0;
    int flag = 0;
    for (i = pos; i < st_len; i++ )
    {
        if (st[i] > '0' && st[i] < '9')
        {
            flag = 1;
            re   = re * 10;
            re   += st[i] - '0';
        }
        else if (flag)
            return re;
        else if (st[i] == 0)
            break;
    }

    return re;
}

void WFG_ini ()
{
    wfg_w    = malloc (sizeof(double) * number_variable + number_objective);
    temp     = malloc (sizeof(double) * number_objective);
    wfg_temp = malloc (sizeof(double) * (number_variable + number_objective));
}

void WFG_free ()
{
    free (wfg_temp);
    free (temp);
    free (wfg_w);
}

int WFG_normalise (double *z, int z_size, double *result)
{
    int i;
    double bound;

    for (i = 0; i < z_size; i++)
    {
        bound     = 2.0 * (i + 1);
        result[i] =  z[i] / bound;
    }
    return z_size;
}

void calculate_x (double *x, double *result, int size)
{
    int i;
    double val = x[size - 1];

    if (val < 1)
        val = 1;

    result[0]        = x[0];
    result[size - 1] = x[size - 1];

    for (i = 1; i < size - 1; i++)
        result[i] = (x[i] - 0.5) * val + 0.5;
}

void calculate_f (double D, double x, double *h, int size, double *result)
{
    int i;
    int S = 0;

    for (i = 0; i < size; i++)
    {
        S = S + 2;
        result[i] = (D * x + S * h[i]);
    }
}


/*
 * Shape Function:
 * The following functions define the shape of the PF.
 * */
double linear (double *x, int M, int m)
{
    int i;
    double result = 1.0;

    for (i = 1; i <= M - m; i++)
        result *= x[i - 1];

    if (m != 1)
        result *= 1 - x[M-m];

    if (result > 1) result = 1;
    if (result < 0) result = 0;

    return result;
}

double convex (double *x, int x_size, int m)
{
    int i;
    double result = 1.0;
    //printf("Into convex... (%d/%d)\n",x_size,m);
    for (i = 1; i <= x_size - m; i++)
        result *= 1.0 - cos(x[i - 1] * PI / 2.0);

    if (m != 1)
        result *= 1.0 - sin(x[x_size - m] * PI / 2.0);

    if (result > 1) result = 1;
    if (result < 0) result = 0;

    return result;
}

double strongconvex (double *x, int x_size, int m)
{
    int i;
    double result = 1.0;
    //printf("Into convex... (%d/%d)\n",x_size,m);
    for (i = 1; i <= x_size - m; i++)
        result *= 1.0 - cos(x[i - 1] * PI / 2.0);

    if (m != 1)
        result *= 1.0 - sin(x[x_size - m] * PI / 2.0);

    if (result > 1) result = 1;
    if (result < 0) result = 0;

    return result;
}
double concave (double *x, int x_size, int m)
{
    int i;
    double result = 1.0;

    for(i = 1; i <= x_size - m; i++)
        result *= sin(x[i - 1] * PI / 2.0);

    if (m != 1)
        result *= cos(x[x_size - m] * PI / 2.0);

    if (result > 1) result = 1;
    if (result < 0) result = 0;

    return result;
}

double mixed (double *x, int A, double alpha)
{
    double tmp    = 2.0 * A * PI;
    double result = pow (1.0 - x[0] - cos (tmp * x[0] + PI / 2.0) / tmp, alpha);

    if (result > 1) result = 1;
    if (result < 0) result = 0;

    return result;
}

double disc (double *x, int A, double alpha, double beta)
{
    double tmp1   = A * pow (x[0], beta) * PI;
    double result = 1.0 - pow (x[0], alpha) * pow (cos(tmp1), 2.0);
    if( result >1 ) result = 1;
    if( result < 0) result = 0;

    return result;
}

/*
 * Transform Fuction:
 * The following functions defines the functions that control the landscape of the search space.
 * */
double b_poly (double y, double alpha)
{
    double result = pow (y, alpha);

    if (result > 1) result = 1;
    if (result < 0) result = 0;

    return result;
}

double min (double a, double b)
{
    if (a > b) return b;

    return a;
}

double b_flat (double y, double A,double B, double C)
{
    double tmp1   = min (0.0, floor(y - B)) * A * (B - y) / B;
    double tmp2   = min (0.0, floor(C - y)) * (1.0 - A) * (y - C) / (1.0 - C);
    double result = A + tmp1 - tmp2;

    if (result > 1) result = 1;
    if (result < 0) result = 0;

    return result;
}

double b_param (double y, double u, double A, double B, double C)
{
    double v      = A - (1.0 - 2.0 * u) * fabs (floor (0.5 - u) + A);
    double result = pow (y, B + (C - B) * v);

    if (result > 1) result = 1;
    if (result < 0) result = 0;

    return result;
}

double s_linear (double y, double A)
{
    double result = fabs (y - A) / fabs (floor(A - y) + A);

    if (result > 1) result = 1;
    if (result < 0) result = 0;

    return result;
}

double s_decept (double y, double A, double B, double C)
{
    double tmp1   = floor (y - A + B) * (1.0 - C + (A - B) / B) / (A - B);
    double tmp2   = floor (A + B - y) * (1.0 - C + (1.0 - A - B) / B) / ( 1.0 - A - B);
    double result = 1.0 + (fabs (y - A) - B) * (tmp1 + tmp2 + 1.0 / B);

    if (result > 1) result = 1;
    if (result < 0) result = 0;

    return result;
}

double s_multi (double y, int A, double B, double C)
{
    double tmp1   = fabs (y - C) / (2.0 * (floor (C - y) + C));
    double tmp2   = (4.0 * A + 2.0) * PI * (0.5 - tmp1);
    double result = (1.0 + cos(tmp2) + 4.0 * B * pow (tmp1, 2.0)) / (B + 2.0);

    if (result > 1) result = 1;
    if (result < 0) result = 0;

    return  result;
}

double r_sum (double *y, int y_size, double *w, int w_size)
{
    int i;
    double result;
    double numerator   = 0.0;
    double denominator = 0.0;

    for (i = 0; i < y_size; i++)
    {
        numerator   += w[i] * y[i];
        denominator += w[i];
    }

    result = numerator / denominator;
    if (result > 1) result = 1;
    if (result < 0) result = 0;

    return result;
}

double r_nonsep (double *y, int y_size, const int A)
{
    int i, j;
    double result;
    double numerator = 0.0;
    for (i = 0; i < y_size; i++)
    {
        numerator += y[i];

        for (j = 0; j <= A - 2; j++)
            numerator += fabs (y[i] - y[(i + j + 1) % y_size]);
    }

    const double tmp = ceil (A / 2.0);
    const double denominator = y_size * tmp * (1.0 + 2.0 * A - 2.0 * tmp) / A;

    result = numerator / denominator;
    if (result > 1) result = 1;
    if (result < 0) result = 0;

    return result;
}

/*
 * Transform Functions Set:
 * This defines the tranform function used by different WFG instances.
 * */
int WFG1_t1 (double *y, int y_size, int k, double *result)
{
    int i;
    for (i = 0; i < k; i++)
        result[i] = y [i];

    for (i = k; i < y_size; i++)
        result[i]= s_linear (y[i], 0.35);

    return y_size;
}

int WFG1_t2 (double *y, int y_size, int k, double *result)
{
    int i;

    for (i = 0; i < k; i++)
        result[i] = y[i];

    for (i = k; i < y_size; i++)
        result[i] = b_flat (y[i], 0.8, 0.75, 0.85);

    return y_size;
}

int WFG1_t3 (double* y, int y_size, double *result)
{
    int i;

    for (i = 0; i < y_size; i++)
    {

            y[i] = (long)(y[i] * 1e8)*1e-8 ;
        result[i] = b_poly (y[i], 0.02);
    }


    return y_size;
}

int WFG1_t4 (double *y, int y_size, int k, int M, double *result)
{
    int i;
    int head, tail;
    for (i = 1; i <= y_size; i++)
        wfg_w[i - 1] = 2.0 * i;

    for (i = 1; i <= M - 1; i++)
    {
        head = (i - 1) * k / (M - 1);
        tail = i * k / (M - 1);
        temp[i - 1] = r_sum (y + head, tail - head, wfg_w + head, tail -head);
    }

    temp[M - 1] = r_sum (y + k, y_size - k, wfg_w + k, y_size - k);

    for (i = 0; i < M ; i++)
        result[i] = temp[i];

    return M;
}

int WFG2_t2 (double *y, int y_size, int k, double *result)
{
    int i;
    const int l = y_size - k;

    for (i = 0; i < k; i++)
        result[i] = y[i];

    for (i = k + 1; i <= k + l / 2; i++)
    {
        const int head = k + 2 * ( i - k ) - 2;
        const int tail = k + 2 * ( i - k );

        result[i] = r_nonsep (y + head, tail - head, 2);
    }

    return k + l / 2 + 1;
}

int WFG2_t3 (double *y, int y_size, int k, int M, double *result)
{
    int i;

    for (i = 1; i <= y_size; i++)
        wfg_w[i - 1] = 1.0;

    for (i = 1; i <= M - 1; i++)
    {
        const int head = (i - 1) * k / (M - 1);
        const int tail = i * k / (M - 1);
        result[i - 1]  = r_sum (y + head, tail - head, wfg_w + head, tail - head);
    }

    result[M - 1] = r_sum (y + k, y_size - k, wfg_w + k, y_size - k);

    return M;
}

int WFG4_t1 (double *y, int y_size, double * result)
{
    int i;

    for (i = 0; i < y_size; i++)
        result[i] = s_multi (y[i], 30, 10, 0.35);
    return y_size;
}


int WFG5_t1 (double *y , int y_size, double *result)
{
    int i;

    for (i = 0; i < y_size; i++)
        result[i] = s_decept (y[i], 0.35, 0.001, 0.05);

    return y_size;
}

int WFG6_t2 (double *y, int y_size, int k, const int M, double *result)
{
    int i;

    for (i = 1; i <= M - 1; i++)
    {
        const int head = (i - 1) * k / (M - 1);
        const int tail = i * k / (M - 1);

        result[i - 1] = (r_nonsep( y+head,tail-head, k/( M-1 ) ) );
    }

    result[M - 1] = (r_nonsep (y + k, y_size - k, y_size - k));

    return M;
}

int WFG7_t1 (double *y, int y_size, int k, double *result)
{
    int i;
    double u;

    for (i = 0; i < y_size ;i++)
        wfg_w[i] = 1.0;

    for (i = 0; i < k; i++)
    {
        u         = r_sum (y + i + 1, y_size - (i + 1), wfg_w + i + 1, y_size - (i + 1));
        result[i] = (b_param (y[i], u, 0.98 / 49.98, 0.02, 50));
    }

    for (i = k; i < y_size; i++)
        result[i] = (y[i]);

    return y_size;
}

int WFG8_t1 (double *y, int y_size, int k, double *result)
{
    int i;
    double u;

    for (i = 0; i < y_size ;i++ )
        wfg_w[i] = 1.0;

    for( i = 0; i < k; i++ )
        result[i] = ( y[i] );

    for( i = k; i < y_size; i++ )
    {
        u         = r_sum (y, i, wfg_w, i);
        result[i] = b_param (y[i], u, 0.98 / 49.98, 0.02, 50);
    }

    return y_size;
}

int WFG9_t1 (double *y, int y_size, double *result)
{
    int i;
    double u;

    for (i = 0; i < y_size; i++)
        wfg_w[i] = 1.0;

    for (i = 0; i < y_size-1; i++)
    {
        u = r_sum (y + i + 1, y_size - (i + 1), wfg_w, y_size - (i + 1));
        result[i] = b_param (y[i], u, 0.98 / 49.98, 0.02, 50);
    }
    result[y_size - 1] = y[y_size - 1];

    return y_size;
}

int WFG9_t2 (double *y, int y_size, int k, double *result)
{
    int i;

    for (i = 0; i < k; i++)
        result[i] = s_decept (y[i], 0.35, 0.001, 0.05);

    for (i = k; i < y_size; i++)
        result[i] = s_multi (y[i], 30, 95, 0.35);

    return y_size;
}

void WFG1_shape (double *y, int size, double *result)
{
    int i;

    calculate_x (y, temp, size);

    for (i = 1; i <= size - 1; i++)
        result[i - 1] = convex (temp, size, i);

    result[size - 1] = mixed (temp, 5, 1.0);

    calculate_f (1.0, temp[size - 1], result, size, result);
}

void WFG2_shape (double *y, int size, double *result)
{
    int i;

    calculate_x (y, temp, size);

    for (i = 1;i <= size - 1; i++)
        result[i - 1] = convex (temp, size, i);

    result[size - 1] = disc (temp, 5, 1.0, 1.0);

    calculate_f (1.0, temp[size - 1], result, size, result);
}

void WFG3_shape (double *y, int y_size, double *result)
{
    int i;

    calculate_x (y, temp, y_size);

    for (i = 1; i <= y_size; i++)
        result[i - 1] = linear (temp, y_size, i);

    calculate_f (1.0, temp[y_size - 1], result, y_size, result);
}

void WFG4_shape (double *y, int y_size, double *result)
{
    int i;

    calculate_x (y, temp, y_size);
    for (i = 1; i <= y_size; i++)
        result[i - 1] = concave (temp, y_size, i);

    calculate_f (1.0, temp[y_size - 1], result, y_size, result);
}

void WFG42_shape (double *y, int y_size, double *result)
{
    int i;

    calculate_x(y, temp, y_size);
    for (i = 1; i <= y_size; i++)
        result[i - 1] = convex (temp, y_size, i);

    calculate_f (1.0, temp[y_size - 1], result, y_size, result);
}

void WFG43_shape (double *y, int y_size, double *result)
{
    int i;

    calculate_x (y, temp, y_size);
    for (i = 1; i <= y_size; i++)
        result[i - 1] = pow (concave (temp, y_size, i), 0.25);

    calculate_f (1.0, temp[y_size - 1], result, y_size, result);
}

void WFG44_shape (double *y, int y_size, double *result)
{
    int i;

    calculate_x (y, temp, y_size);
    for (i = 1; i <= y_size; i++)
        result[i - 1] = pow (convex (temp, y_size, i), 2);

    calculate_f (1.0, temp[y_size - 1], result, y_size, result);
}

void WFG45_shape (double *y, int y_size, double *result)
{
    int i;

    calculate_x (y, temp, y_size);
    for (i = 1; i <= y_size - 1; i++)
        result[i - 1] = convex (temp, y_size, i);
    result[y_size - 1] = mixed (temp, 2, 1.0);

    calculate_f (1.0, temp[y_size - 1], result, y_size, result);
}

void WFG46_shape (double *y, int y_size, double *result)
{
    int i;

    calculate_x (y, temp, y_size);
    for (i = 1; i <= y_size; i++)
        result[i - 1] = linear (temp, number_objective, i);

    calculate_f (1.0, temp[y_size - 1], result, y_size, result);
}

void WFG47_shape (double *y, int y_size, double *result)
{
    int i;

    calculate_x (y, temp, y_size);
    for (i = 1; i <= y_size - 1; i++)
        result[i - 1] = concave (temp, number_objective, i);
    result[y_size - 1] = disc(temp, 2, 0.5, 0.5);

    calculate_f (1.0, temp[y_size - 1], result, y_size, result);
}

void WFG48_shape (double *y, int y_size, double *result)
{
    int i;

    calculate_x (y, temp, y_size);
    for (i = 1; i <= y_size - 1; i++)
        result[i - 1] = convex (temp, number_objective, i);
    result[y_size - 1] = disc (temp, 2, 0.5, 0.5);

    calculate_f (1.0, temp[y_size - 1], result, y_size, result);
}

void WFG21_shape (double *y, int size, double *result)
{
    int i;

    calculate_x (y, temp, size);

    for (i = 1;i <= size - 1; i++)
        result[i - 1] = convex (temp, size, i);

    result[size - 1] = disc (temp, 10, 1.0, 1.0);

    calculate_f (1.0, temp[size - 1], result, size, result);
}

void WFG22_shape (double *y, int size, double *result)
{
    int i;

    calculate_x (y, temp, size);

    for (i = 1;i <= size - 1; i++)
        result[i - 1] = convex (temp, size, i);

    result[size - 1] = disc (temp, 5, 5.0, 1.0);

    calculate_f (1.0, temp[size - 1], result, size, result);
}

void WFG23_shape (double *y, int size, double *result)
{
    int i;

    calculate_x (y, temp, size);

    for (i = 1;i <= size - 1; i++)
        result[i - 1] = convex (temp, size, i);

    result[size - 1] = disc (temp, 5, 1.0, 5.0);

    calculate_f (1.0, temp[size - 1], result, size, result);
}

void WFG24_shape (double *y, int size, double *result)
{
    int i;

    calculate_x (y, temp, size);

    for (i = 1;i <= size - 1; i++)
        result[i - 1] = convex (temp, size, i);

    result[size - 1] = disc (temp, 5, 5.0, 5.0);

    calculate_f (1.0, temp[size - 1], result, size, result);
}
