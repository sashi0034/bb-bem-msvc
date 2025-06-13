#ifndef BICGSTAB_CUDA_H
#define BICGSTAB_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void serial_bicgstab_cuda(
    int batch_size,
    int dim,
    double** A, // in [dim][dim]
    double** b, // in [batch][dim]
    double** x, // inout [batch][dim]
    double tor,
    int max_steps
);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // BICGSTAB_CUDA_H
