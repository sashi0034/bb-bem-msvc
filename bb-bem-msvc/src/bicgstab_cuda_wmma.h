#if 0 // We don't need this file in this branch

#ifndef BICGSTAB_CUDA_WMMA_H
#define BICGSTAB_CUDA_WMMA_H

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void bicgstab_cuda_wmma(
    int batch,
    int dim,
    double** A /* in [dim][dim] */,
    double** b /* in [dim][batch] */,
    double** x /* out [dim][batch] */,
    double tor,
    int max_steps
);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // BICGSTAB_CUDA_WMMA_H

#endif
