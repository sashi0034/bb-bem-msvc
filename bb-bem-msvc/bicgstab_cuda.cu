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
__global__ void kernel_matvec(
    int dim,
    int batch,
    const double* __restrict__ mat,
    const double* __restrict__ P,
    double* __restrict__ Q
) {
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
__global__ void kernel_residual(int dim, int batch,
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
__global__ void kernel_dot_product(int dim, int batch,
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
__global__ void kernel_sqrt(int batch, const double* x, double* out) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < batch) out[n] = sqrt(x[n]);
}

__global__ void kernel_mul(int batch, const double* x, const double* y, double* out) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < batch) out[n] = x[n] * y[n];
}

__global__ void kernel_div(int batch, const double* x, const double* y, double* out) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < batch) out[n] = x[n] / y[n];
}

// x < y
static int batch_lt(int batch, const double* x, double y) {
    for (int n = 0; n < batch; ++n) {
        if (x[n] >= y) return 0;
    }

    return 1;
}

// Kernel: p = r + beta * (p - zeta * Ap)
__global__ void kernel_update_p(int dim, int batch,
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
__global__ void kernel_update_t(int dim, int batch,
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
__global__ void kernel_update_x(int dim, int batch,
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
__global__ void kernel_update_r(int dim, int batch,
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
    double** A /* in [dim][dim] */,
    double** b /* in [dim][batch] */,
    double** x /* out [dim][batch] */,
    double tor,
    int max_steps
) {
    const size_t dim_dim_bytes = static_cast<size_t>(dim) * dim * sizeof(double);
    const size_t dim_batch_bytes = static_cast<size_t>(dim) * batch * sizeof(double);
    const size_t batch_bytes = static_cast<size_t>(batch) * sizeof(double);

    // Device buffers
    double *d_A, *d_b, *d_x;
    CUDA_CHECK(cudaMalloc(&d_A, dim_dim_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, dim_batch_bytes));
    CUDA_CHECK(cudaMalloc(&d_x, dim_batch_bytes));

    CUDA_CHECK(cudaMemcpy(d_A, A[0], dim_dim_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b[0], dim_batch_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, x[0], dim_batch_bytes, cudaMemcpyHostToDevice));

    // Work arrays
    double *d_p, *d_r, *d_r0, *d_t, *d_Ap, *d_Akp, *d_kt, *d_Akt, *d_kp;
    cudaMalloc(&d_p, dim_batch_bytes);
    cudaMalloc(&d_r, dim_batch_bytes);
    cudaMalloc(&d_r0, dim_batch_bytes);
    cudaMalloc(&d_t, dim_batch_bytes);
    cudaMalloc(&d_Ap, dim_batch_bytes);
    cudaMalloc(&d_Akp, dim_batch_bytes);
    cudaMalloc(&d_kt, dim_batch_bytes);
    cudaMalloc(&d_Akt, dim_batch_bytes);
    cudaMalloc(&d_kp, dim_batch_bytes);

    double *d_bnorm, *d_rnorm, *d_nom, *d_nom_old, *d_den, *d_alpha, *d_beta, *d_zeta, *d_tmp;
    cudaMalloc(&d_bnorm, batch_bytes);
    cudaMalloc(&d_rnorm, batch_bytes);
    cudaMalloc(&d_nom, batch_bytes);
    cudaMalloc(&d_nom_old, batch_bytes);
    cudaMalloc(&d_den, batch_bytes);
    cudaMalloc(&d_alpha, batch_bytes);
    cudaMalloc(&d_beta, batch_bytes);
    cudaMalloc(&d_zeta, batch_bytes);
    cudaMalloc(&d_tmp, batch_bytes);

    // -----------------------------------------------

    dim3 block2d{16, 16, 1};
    dim3 grid2d((dim + 15) / 16, (batch + 15) / 16, 1);
    int threads1d = 256;
    int blocks1d = (batch + threads1d - 1) / threads1d;

    // bnorm = sqrt(dot_product(dim, b, b));
    kernel_dot_product<<<blocks1d, threads1d>>>(dim, batch, d_b, d_b, d_bnorm);
    kernel_sqrt<<<blocks1d, threads1d>>>(batch, d_bnorm, d_bnorm);

    // r = b - A * x
    kernel_residual<<<grid2d,block2d>>>(dim, batch, d_A, d_x, d_b, d_r);
    cudaMemcpy(d_r0, d_r, dim_batch_bytes, cudaMemcpyDeviceToDevice);
    kernel_dot_product<<<blocks1d, threads1d>>>(dim, batch, d_r, d_r, d_rnorm);
    kernel_sqrt<<<blocks1d, threads1d>>>(batch, d_rnorm, d_rnorm);

    cudaMemset(d_p, 0, dim_batch_bytes);
    cudaMemset(d_alpha, 0, batch_bytes);
    cudaMemset(d_beta, 0, batch_bytes);
    cudaMemset(d_zeta, 0, batch_bytes);

    // if (rnorm / bnorm < tor) { break; } // early exit
    double* tmp = static_cast<double*>(malloc(sizeof(double) * batch)); // <-- Allocation: tmp
    kernel_div<<<blocks1d, threads1d>>>(batch, d_rnorm, d_bnorm, d_tmp);

    cudaMemcpy(tmp, d_tmp, batch_bytes, cudaMemcpyDeviceToHost);
    printf("Original rel res [0] = %20.14e\n", tmp[0]);
    if (batch_lt(batch, tmp, tor)) { goto finalize; }

    // BiCGSTAB iteration 
    for (int step = 1; step <= max_steps; ++step) {
        // matvec(dim, A, p, Ap);
        kernel_matvec<<<grid2d, block2d>>>(dim, batch, d_A, d_p, d_Ap);

        // p[i] = r[i] + beta * (p[i] - zeta * Ap[i]);
        kernel_update_p<<<grid2d, block2d>>>(dim, batch, d_p, d_r, d_p, d_Ap, d_beta, d_zeta);
        cudaMemcpy(d_kp, d_p, dim_batch_bytes, cudaMemcpyDeviceToDevice);

        // matvec(dim, A, kp, Akp);
        kernel_matvec<<<grid2d, block2d>>>(dim, batch, d_A, d_kp, d_Akp);

        // nom = dot_product(dim, r0, r);
        kernel_dot_product<<<blocks1d, threads1d>>>(dim, batch, d_r0, d_r, d_nom);

        // den = dot_product(dim, r0, Akp);
        kernel_dot_product<<<blocks1d, threads1d>>>(dim, batch, d_r0, d_Akp, d_den);

        // alpha = nom / den;
        kernel_div<<<blocks1d, threads1d>>>(batch, d_nom, d_den, d_alpha);

        // nom_old = nom;
        cudaMemcpy(d_nom_old, d_nom, batch_bytes, cudaMemcpyDeviceToDevice);

        // t[i] = r[i] - alpha * Akp[i];
        kernel_update_t<<<grid2d,block2d>>>(dim, batch, d_r, d_Akp, d_alpha, d_t);

        // kt[i] = t[i];
        cudaMemcpy(d_kt, d_t, dim_batch_bytes, cudaMemcpyDeviceToDevice);

        //  matvec(dim, A, kt, Akt);
        kernel_matvec<<<grid2d,block2d>>>(dim, batch, d_A, d_kt, d_Akt);

        // nom = dot_product(dim, Akt, t);
        kernel_dot_product<<<blocks1d, threads1d>>>(dim, batch, d_Akt, d_t, d_nom);

        // den = dot_product(dim, Akt, Akt);
        kernel_dot_product<<<blocks1d, threads1d>>>(dim, batch, d_Akt, d_Akt, d_den);

        // zeta = nom / den;
        kernel_div<<<blocks1d, threads1d>>>(batch, d_nom, d_den, d_zeta);

        // x[i] = x[i] + alpha * kp[i] + zeta * kt[i];
        kernel_update_x<<<grid2d,block2d>>>(dim, batch, d_x, d_kp, d_kt, d_alpha, d_zeta);

        // r[i] = t[i] - zeta * Akt[i];
        kernel_update_r<<<grid2d,block2d>>>(dim, batch, d_t, d_Akt, d_zeta, d_r);

        // beta = alpha/zeta * dot(r0,r) / nom_old
        kernel_dot_product<<<blocks1d, threads1d>>>(dim, batch, d_r0, d_r, d_tmp); // dot_product(dim, r0, r)
        kernel_mul<<<blocks1d, threads1d>>>(batch, d_alpha, d_tmp, d_beta); // alpha * dot_product(dim, r0, r)
        kernel_div<<<blocks1d, threads1d>>>(batch, d_beta, d_zeta, d_beta); // alpha / zeta * dot_product(dim, r0, r)
        kernel_div<<<blocks1d, threads1d>>>(
            batch, d_beta, d_nom_old, d_beta); // alpha / zeta * dot_product(dim, r0, r) / nom_old

        // rnorm and check
        kernel_dot_product<<<blocks1d, threads1d>>>(dim, batch, d_r, d_r, d_rnorm); // dot_product(dim, r, r)
        kernel_sqrt<<<blocks1d, threads1d>>>(batch, d_rnorm, d_rnorm); // sqrt(dot_product(dim, r, r))

        // if (rnorm / bnorm < tor) { break; }
        kernel_div<<<blocks1d, threads1d>>>(batch, d_rnorm, d_bnorm, d_tmp); // TODO

        // if (rnorm / bnorm < tor) { break; }
        cudaMemcpy(tmp, d_tmp, batch_bytes, cudaMemcpyDeviceToHost);
        printf("  Step %d rel res [0] = %20.14e\n", step, tmp[0]);
        if (batch_lt(batch, tmp, tor)) { goto finalize; }
    }

finalize:
    cudaMemcpy(x[0], d_x, dim_batch_bytes, cudaMemcpyDeviceToHost);

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

    free(tmp);
}
