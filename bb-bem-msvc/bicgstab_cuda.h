#ifndef BICGSTAB_CUDA_H
#define BICGSTAB_CUDA_H

void bicgstab_cuda(
    int batch,
    int dim,
    double** A /* in [dim][dim] */,
    double** b /* in [dim][batch] */,
    double** x /* out [dim][batch] */,
    double tor,
    int max_steps
);

#endif // BICGSTAB_CUDA_H
