#ifndef BICGSTAB_CUDA_H
#define BICGSTAB_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void bicgstab_cuda(
    int dim,
    double** A, // in [dim][dim]
    double* b, // in [dim]
    double* x, // inout [dim]
    double tor,
    int max_steps
);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // BICGSTAB_CUDA_H
