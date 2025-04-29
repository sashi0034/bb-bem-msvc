#ifndef BICGSTAB_NAIVE_H
#define BICGSTAB_NAIVE_H

void bicgstab_naive(
    int batch,
    int dim,
    double** A /* [dim][dim] */,
    double** b /* [dim][batch] */,
    double** x /* [dim][batch] */,
    double tor,
    int max_steps
);

#endif // BICGSTAB_NAIVE_H
