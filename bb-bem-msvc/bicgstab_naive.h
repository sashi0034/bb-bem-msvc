#ifndef BICGSTAB_NAIVE_H
#define BICGSTAB_NAIVE_H

void bicgstab_naive(
    int batch,
    int dim,
    double** A /* in [dim][dim] */,
    double** b /* in [dim][batch] */,
    double** x /* out [dim][batch] */,
    double tor,
    int max_steps
);

#endif // BICGSTAB_NAIVE_H
