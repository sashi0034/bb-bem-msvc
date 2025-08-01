﻿#ifndef BICGSTAB_CUDA_CUBLAS_H
#define BICGSTAB_CUDA_CUBLAS_H

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void bicgstab_cuda_cublas(
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

#endif // BICGSTAB_CUDA_CUBLAS_H
