#include"DTLZ.h"

void dtlz71 (double *xreal, double*obj)
{
    int i, k;
    double h, gx;
    double A = 5.0;
    double alpha = 0.0;
    double beta = 1.0;
    gx = 0.0;
    k  = number_variable - number_objective + 1;
    for(i = number_variable - k; i < number_variable; i++)
        gx += xreal[i];
    gx = 1.0 + (9.0 * gx) / k;

    for (i = 0; i < number_objective; i++)
        obj[i] = xreal[i];

    h = 0.0;
    for (i = 0; i < number_objective - 1; i++)
        h += (obj[i] / (1.0 + gx)) * (1.0 + pow(obj[i],alpha) * sin (A * PI * pow(obj[i],beta)));
    h = number_objective - h;

    obj[number_objective - 1] = (1 + gx) * h;

    double m_max,m_min;
    m_min = 2;
    m_max = 2*number_objective;
    //obj[number_objective - 1] = (obj[number_objective - 1] - m_min)/(m_max-m_min);
}

void dtlz73 (double *xreal, double*obj)
{
    int i, k;
    double h, gx;
    double A = 3.0;
    double alpha = 3.0;
    double beta = 1.0;
    gx = 0.0;
    k  = number_variable - number_objective + 1;
    for(i = number_variable - k; i < number_variable; i++)
        gx += xreal[i];
    gx = 1.0 + (9.0 * gx) / k;

    for (i = 0; i < number_objective; i++)
        obj[i] = xreal[i];

    h = 0.0;
    for (i = 0; i < number_objective - 1; i++)
        h += (obj[i] / (1.0 + gx)) * (1.0 + pow(obj[i],alpha) * sin (A * PI * pow(obj[i],beta)));
    h = number_objective - h;

    obj[number_objective - 1] = (1 + gx) * h;

    double m_max,m_min;
    m_min = 2;
    m_max = 2*number_objective;
    //obj[number_objective - 1] = (obj[number_objective - 1] - m_min)/(m_max-m_min);
}

void dtlz72 (double *xreal, double*obj)
{
    int i, k;
    double h, gx;
    double A = 3.0;
    double alpha = 0.0;
    double beta = 2.0;
    gx = 0.0;
    k  = number_variable - number_objective + 1;
    for(i = number_variable - k; i < number_variable; i++)
        gx += xreal[i];
    gx = 1.0 + (9.0 * gx) / k;

    for (i = 0; i < number_objective; i++)
        obj[i] = xreal[i];

    h = 0.0;
    for (i = 0; i < number_objective - 1; i++)
        h += (obj[i] / (1.0 + gx)) * (1.0 + pow(obj[i],alpha) * sin (A * PI * pow(obj[i],beta)));
    h = number_objective - h;

    obj[number_objective - 1] = (1 + gx) * h;

    double m_max,m_min;
    m_min = 2;
    m_max = 2*number_objective;
    //obj[number_objective - 1] = (obj[number_objective - 1] - m_min)/(m_max-m_min);
}

