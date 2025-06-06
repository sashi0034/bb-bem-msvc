#ifndef BICGSTAB_NAIVE_H
#define BICGSTAB_NAIVE_H

void bicgstab_naive(
    int dim,
    double** A /* in [dim][dim] */,
    double* b /* in [dim] */,
    double* x /* out [dim] */,
    double tor,
    int max_steps
);

#endif // BICGSTAB_NAIVE_H
