#ifndef BICGSTAB_NAIVE_H
#define BICGSTAB_NAIVE_H

void bicgstab_naive(
    int batch,
    int dim,
    double** A /* in [dim][dim] */,
    double** b /* in [batch][dim] */,
    double** x /* out [batch][dim] */,
    double tor,
    int max_steps
);

#endif // BICGSTAB_NAIVE_H
