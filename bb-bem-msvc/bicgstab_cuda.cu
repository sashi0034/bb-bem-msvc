#include <cuda_runtime.h>
#include <stdio.h>

#include "bicgstab_cuda.h"

#define CUDA_CHECK(err) do { \
    cudaError_t _e = (err); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA Error %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_e)); \
        return; \
    } \
} while (0)

// Kernel: Q[row, n] = sum_col A[row, col] * P[col, n]
__global__ void batch_matvec_kernel(int dim, int batch,
                                    const double* __restrict__ mat,
                                    const double* __restrict__ P,
                                    double* __restrict__ Q) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < dim && n < batch) {
        double sum = 0.0;
        const double* Arow = mat + row * dim;
        const double* Pcol = P + n;
        for (int col = 0; col < dim; ++col) {
            sum += Arow[col] * Pcol[col * batch];
        }
        Q[row * batch + n] = sum;
    }
}

// Kernel: R = B - A * X
__global__ void batch_residual_kernel(int dim, int batch,
                                      const double* __restrict__ A,
                                      const double* __restrict__ X,
                                      const double* __restrict__ B,
                                      double* __restrict__ R) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < dim && n < batch) {
        double val = B[row * batch + n];
        const double* Arow = A + row * dim;
        const double* Xcol = X + n;
        for (int col = 0; col < dim; ++col) {
            val -= Arow[col] * Xcol[col * batch];
        }
        R[row * batch + n] = val;
    }
}

// Kernel: out[n] = dot( X[:,n], Y[:,n] )
__global__ void batch_dot_product_kernel(int dim, int batch,
                                         const double* __restrict__ X,
                                         const double* __restrict__ Y,
                                         double* __restrict__ out) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < batch) {
        double sum = 0.0;
        const double* Xcol = X + n;
        const double* Ycol = Y + n;
        for (int i = 0; i < dim; ++i) {
            sum += Xcol[i * batch] * Ycol[i * batch];
        }
        out[n] = sum;
    }
}

// Elementwise kernels
__global__ void batch_sqrt_kernel(int batch, const double* x, double* out) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < batch) out[n] = sqrt(x[n]);
}

__global__ void batch_mul_kernel(int batch, const double* x, const double* y, double* out) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < batch) out[n] = x[n] * y[n];
}

__global__ void batch_div_kernel(int batch, const double* x, const double* y, double* out) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < batch) out[n] = x[n] / y[n];
}

// Kernel: p = r + beta * (p - zeta * Ap)
__global__ void update_p_kernel(int dim, int batch,
                                double* __restrict__ out,
                                const double* __restrict__ r,
                                const double* __restrict__ p,
                                const double* __restrict__ Ap,
                                const double* __restrict__ beta,
                                const double* __restrict__ zeta) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < dim && n < batch) {
        int idx = row * batch + n;
        out[idx] = r[idx] + beta[n] * (p[idx] - zeta[n] * Ap[idx]);
    }
}

// Kernel: t = r - alpha * Akp
__global__ void update_t_kernel(int dim, int batch,
                                const double* __restrict__ r,
                                const double* __restrict__ Akp,
                                const double* __restrict__ alpha,
                                double* __restrict__ t) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < dim && n < batch) {
        int idx = row * batch + n;
        t[idx] = r[idx] - alpha[n] * Akp[idx];
    }
}

// Kernel: x += alpha * kp + zeta * kt
__global__ void update_x_kernel(int dim, int batch,
                                double* __restrict__ x,
                                const double* __restrict__ kp,
                                const double* __restrict__ kt,
                                const double* __restrict__ alpha,
                                const double* __restrict__ zeta) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < dim && n < batch) {
        int idx = row * batch + n;
        x[idx] += alpha[n] * kp[idx] + zeta[n] * kt[idx];
    }
}

// Kernel: r = t - zeta * Akt
__global__ void update_r_kernel(int dim, int batch,
                                const double* __restrict__ t,
                                const double* __restrict__ Akt,
                                const double* __restrict__ zeta,
                                double* __restrict__ r) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < dim && n < batch) {
        int idx = row * batch + n;
        r[idx] = t[idx] - zeta[n] * Akt[idx];
    }
}

void bicgstab_cuda(
    int batch,
    int dim,
    double** A,
    double** b,
    double** x,
    double tor,
    int max_steps
) {
    size_t mat_size = (size_t)dim * dim * sizeof(double);
    size_t batch_size = (size_t)dim * batch * sizeof(double);
    size_t vec_size = (size_t)batch * sizeof(double);

    // Device buffers
    double *d_A, *d_b, *d_x;
    CUDA_CHECK(cudaMalloc(&d_A, mat_size));
    CUDA_CHECK(cudaMalloc(&d_b, batch_size));
    CUDA_CHECK(cudaMalloc(&d_x, batch_size));
    CUDA_CHECK(cudaMemcpy(d_A, A[0], mat_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b[0], batch_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, x[0], batch_size, cudaMemcpyHostToDevice));

    // Work arrays
    double *d_p, *d_r, *d_r0, *d_t;
    double *d_Ap, *d_Akp, *d_kt, *d_Akt, *d_kp;
    double *d_bnorm, *d_rnorm, *d_nom, *d_nom_old, *d_den;
    double *d_alpha, *d_beta, *d_zeta, *d_tmp;
    cudaMalloc(&d_p, batch_size);
    cudaMalloc(&d_r, batch_size);
    cudaMalloc(&d_r0, batch_size);
    cudaMalloc(&d_t, batch_size);
    cudaMalloc(&d_Ap, batch_size);
    cudaMalloc(&d_Akp, batch_size);
    cudaMalloc(&d_kt, batch_size);
    cudaMalloc(&d_Akt, batch_size);
    cudaMalloc(&d_kp, batch_size);
    cudaMalloc(&d_bnorm, vec_size);
    cudaMalloc(&d_rnorm, vec_size);
    cudaMalloc(&d_nom, vec_size);
    cudaMalloc(&d_nom_old, vec_size);
    cudaMalloc(&d_den, vec_size);
    cudaMalloc(&d_alpha, vec_size);
    cudaMalloc(&d_beta, vec_size);
    cudaMalloc(&d_zeta, vec_size);
    cudaMalloc(&d_tmp, vec_size);

    dim3 block2d(16, 16);
    dim3 grid2d((dim + 15) / 16, (batch + 15) / 16);
    int threads1d = 256;
    int blocks1d = (batch + threads1d - 1) / threads1d;

    // bnorm = sqrt(dot(b,b))
    batch_dot_product_kernel<<<blocks1d,threads1d>>>(dim, batch, d_b, d_b, d_bnorm);
    batch_sqrt_kernel <<<blocks1d,threads1d>>>(batch, d_bnorm, d_bnorm);

    // r = b - A*x
    batch_residual_kernel<<<grid2d,block2d>>>(dim, batch, d_A, d_x, d_b, d_r);
    cudaMemcpy(d_r0, d_r, batch_size, cudaMemcpyDeviceToDevice);
    batch_dot_product_kernel<<<blocks1d,threads1d>>>(dim, batch, d_r, d_r, d_rnorm);
    batch_sqrt_kernel <<<blocks1d,threads1d>>>(batch, d_rnorm, d_rnorm);

    double h_bnorm0, h_rnorm0;
    cudaMemcpy(&h_bnorm0, d_bnorm + 0, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_rnorm0, d_rnorm + 0, sizeof(double), cudaMemcpyDeviceToHost);
    printf("Original rel res [0] = %20.14e\n", h_rnorm0 / h_bnorm0);

    cudaMemset(d_p, 0, batch_size);
    cudaMemset(d_alpha, 0, vec_size);
    cudaMemset(d_beta, 0, vec_size);
    cudaMemset(d_zeta, 0, vec_size);

    // early exit
    batch_div_kernel<<<blocks1d,threads1d>>>(batch, d_rnorm, d_bnorm, d_tmp);
    cudaMemcpy(&h_rnorm0, d_tmp + 0, sizeof(double), cudaMemcpyDeviceToHost);
    if (h_rnorm0 < tor) goto finalize;

    for (int step = 1; step <= max_steps; ++step) {
        batch_matvec_kernel <<<grid2d,block2d>>>(dim, batch, d_A, d_p, d_Ap);
        update_p_kernel <<<grid2d,block2d>>>(dim, batch, d_p, d_r, d_p, d_Ap, d_beta, d_zeta);
        cudaMemcpy(d_kp, d_p, batch_size, cudaMemcpyDeviceToDevice);
        batch_matvec_kernel <<<grid2d,block2d>>>(dim, batch, d_A, d_kp, d_Akp);
        batch_dot_product_kernel<<<blocks1d,threads1d>>>(dim, batch, d_r0, d_r, d_nom);
        batch_dot_product_kernel<<<blocks1d,threads1d>>>(dim, batch, d_r0, d_Akp, d_den);
        batch_div_kernel <<<blocks1d,threads1d>>>(batch, d_nom, d_den, d_alpha);
        cudaMemcpy(d_nom_old, d_nom, vec_size, cudaMemcpyDeviceToDevice);

        update_t_kernel <<<grid2d,block2d>>>(dim, batch, d_r, d_Akp, d_alpha, d_t);
        cudaMemcpy(d_kt, d_t, batch_size, cudaMemcpyDeviceToDevice);
        batch_matvec_kernel <<<grid2d,block2d>>>(dim, batch, d_A, d_kt, d_Akt);
        batch_dot_product_kernel<<<blocks1d,threads1d>>>(dim, batch, d_Akt, d_t, d_nom);
        batch_dot_product_kernel<<<blocks1d,threads1d>>>(dim, batch, d_Akt, d_Akt, d_den);
        batch_div_kernel <<<blocks1d,threads1d>>>(batch, d_nom, d_den, d_zeta);

        update_x_kernel <<<grid2d,block2d>>>(dim, batch, d_x, d_kp, d_kt, d_alpha, d_zeta);
        update_r_kernel <<<grid2d,block2d>>>(dim, batch, d_t, d_Akt, d_zeta, d_r);

        // beta = alpha/zeta * dot(r0,r) / nom_old
        batch_dot_product_kernel<<<blocks1d,threads1d>>>(dim, batch, d_r0, d_r, d_tmp);
        batch_mul_kernel <<<blocks1d,threads1d>>>(batch, d_alpha, d_tmp, d_beta);
        batch_div_kernel <<<blocks1d,threads1d>>>(batch, d_beta, d_zeta, d_beta);
        batch_div_kernel <<<blocks1d,threads1d>>>(batch, d_beta, d_nom_old, d_beta);

        // rnorm and check
        batch_dot_product_kernel<<<blocks1d,threads1d>>>(dim, batch, d_r, d_r, d_rnorm);
        batch_sqrt_kernel <<<blocks1d,threads1d>>>(batch, d_rnorm, d_rnorm);
        batch_div_kernel <<<blocks1d,threads1d>>>(batch, d_rnorm, d_bnorm, d_tmp);
        cudaMemcpy(&h_rnorm0, d_tmp + 0, sizeof(double), cudaMemcpyDeviceToHost);
        printf("  Step %d rel res [0] = %20.14e\n", step, h_rnorm0);
        if (h_rnorm0 < tor) break;
    }

finalize:
    cudaMemcpy(x[0], d_x, batch_size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_p);
    cudaFree(d_r);
    cudaFree(d_r0);
    cudaFree(d_t);
    cudaFree(d_Ap);
    cudaFree(d_Akp);
    cudaFree(d_kt);
    cudaFree(d_Akt);
    cudaFree(d_kp);
    cudaFree(d_bnorm);
    cudaFree(d_rnorm);
    cudaFree(d_nom);
    cudaFree(d_nom_old);
    cudaFree(d_den);
    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_zeta);
    cudaFree(d_tmp);
}
