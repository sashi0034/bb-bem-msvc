#include <mma.h>
using namespace nvcuda;

#include <stdio.h>

#include "bicgstab_cuda_wmma.h"

#define CUDA_CHECK(err) do { \
    cudaError_t _e = (err); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA Error %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_e)); \
        return; \
    } \
} while (0)

static void batch_matvec(
    int batch,
    int dim,
    double** mat /* [dim][dim] */,
    double** P /* [dim][batch] */,
    double** Q /* out [dim][batch] */
) {
    for (int row = 0; row < dim; ++row) {
        for (int n = 0; n < batch; ++n) {
            Q[row][n] = 0.0;
        }
    }

    for (int row = 0; row < dim; ++row) {
        for (int col = 0; col < dim; ++col) {
            for (int n = 0; n < batch; ++n) {
                Q[row][n] += mat[row][col] * P[col][n];
            }
        }
    }
}

// Kernel: Q[n, row] = sum_col A[row, col] * P[n, col]
__global__ static void kernel_matvec(
    int batch,
    int dim,
    const double* __restrict__ mat /* [dim * dim] */,
    const double* __restrict__ P /* [batch * dim] */,
    double* __restrict__ Q /* out [batch * dim] */
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < dim && n < batch) {
        double sum = 0.0;
        const double* Arow = mat + row * dim;
        const double* Prow = P + n * dim;
        for (int col = 0; col < dim; ++col) {
            sum += Arow[col] * Prow[col];
        }

        Q[n * dim + row] = sum;
    }
}

static constexpr int WMMA_M = 8;
static constexpr int WMMA_N = 8;
static constexpr int WMMA_K = 4;

// Kernel: Q[row, n] = sum_col A[row, col] * P[col, n]
// A is [dim][dim], P/Q are now [batch][dim] in row-major
__global__ void wmma_matvec(
    int batch,
    int dim,
    const double* __restrict__ A, /* [dim * dim] */
    const double* __restrict__ P, /* [batch * dim] */
    double* __restrict__ Q /* out [batch * dim] */
) {
    const int block_row = blockIdx.x * WMMA_M;
    const int block_col = blockIdx.y * WMMA_N;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, double> c_frag;
    wmma::fill_fragment(c_frag, 0.0);

    int num_k_tiles = dim / WMMA_K;
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, double, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, double, wmma::col_major> b_frag;

        // A tile pointer: (block_row, k_tile * WMMA_K)
        const double* a_tile_ptr = A + (block_row * dim) + (k_tile * WMMA_K);
        // P tile pointer in transposed layout: (row_new = block_col, col_new = k_tile * WMMA_K)
        const double* b_tile_ptr = P + (block_col * dim) + (k_tile * WMMA_K);

        wmma::load_matrix_sync(a_frag, a_tile_ptr, dim);
        wmma::load_matrix_sync(b_frag, b_tile_ptr, dim);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Q tile pointer in transposed layout: (row_new = block_col, col_new = block_row)
    double* c_tile_ptr = Q + (block_col * dim) + (block_row);
    wmma::store_matrix_sync(c_tile_ptr, c_frag, dim, wmma::mem_col_major);
}

static void launch_wmma_matvec(
    int batch,
    int dim,
    const double* d_A, /* [dim * dim] */
    const double* d_P, /* [batch * dim] */
    double* d_Q /* out [batch * dim] */
) {
    dim3 grid(dim / WMMA_M, batch / WMMA_N);
    dim3 block(32, 1, 1); // one warp per tile

    wmma_matvec<<<grid, block>>>(batch, dim, d_A, d_P, d_Q);
}

// Kernel: R = B - A * X
__global__ static void kernel_residual(
    int batch,
    int dim,
    const double* __restrict__ A /* [dim * dim] */,
    const double* __restrict__ X /* [batch * dim] */,
    const double* __restrict__ B /* [batch * dim] */,
    double* __restrict__ R /* out [batch * dim] */
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < dim && n < batch) {
        double val = B[n * dim + row];
        const double* Arow = A + row * dim;
        const double* Xrow = X + n * dim;
        for (int col = 0; col < dim; ++col) {
            val -= Arow[col] * Xrow[col];
        }

        R[n * dim + row] = val;
    }
}

// Kernel: out[n] = dot( X[:,n], Y[:,n] )
__global__ static void kernel_dot_product(
    int batch,
    int dim,
    const double* __restrict__ X /* [batch * dim] */,
    const double* __restrict__ Y /* [batch * dim] */,
    double* __restrict__ out /* out [batch] */
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < batch) {
        double sum = 0.0;
        const double* Xrow = X + n * dim;
        const double* Yrow = Y + n * dim;
        for (int i = 0; i < dim; ++i) {
            sum += Xrow[i] * Yrow[i];
        }

        out[n] = sum;
    }
}

// Elementwise kernels
__global__ static void kernel_sqrt(
    int batch,
    const double* x /* [batch] */,
    double* out /* out [batch] */
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < batch) out[n] = sqrt(x[n]);
}

__global__ static void kernel_mul(
    int batch,
    const double* x /* [batch] */,
    const double* y /* [batch] */,
    double* out /* out [batch] */
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < batch) out[n] = x[n] * y[n];
}

__global__ static void kernel_div(
    int batch,
    const double* x /* [batch] */,
    const double* y /* [batch] */,
    double* out /* out [batch] */
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
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
__global__ static void kernel_update_p(
    int batch,
    int dim,
    double* __restrict__ out /* out [batch * dim] */,
    const double* __restrict__ r /* [batch * dim] */,
    const double* __restrict__ p /* [batch * dim] */,
    const double* __restrict__ Ap /* [batch * dim] */,
    const double* __restrict__ beta /* [batch] */,
    const double* __restrict__ zeta /* [batch] */
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < dim && n < batch) {
        const int idx = n * dim + row;
        out[idx] = r[idx] + beta[n] * (p[idx] - zeta[n] * Ap[idx]);
    }
}

// Kernel: t = r - alpha * Akp
__global__ static void kernel_update_t(
    int batch,
    int dim,
    const double* __restrict__ r /* [batch * dim] */,
    const double* __restrict__ Akp /* [batch * dim] */,
    const double* __restrict__ alpha /* [batch] */,
    double* __restrict__ t /* out [batch * dim] */
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < dim && n < batch) {
        int idx = n * dim + row;
        t[idx] = r[idx] - alpha[n] * Akp[idx];
    }
}

// Kernel: x += alpha * kp + zeta * kt
__global__ static void kernel_update_x(
    int batch,
    int dim,
    double* __restrict__ x /* inout [batch * dim] */,
    const double* __restrict__ kp /* [batch * dim] */,
    const double* __restrict__ kt /* [batch * dim] */,
    const double* __restrict__ alpha /* [batch] */,
    const double* __restrict__ zeta /* [batch] */
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < dim && n < batch) {
        int idx = n * dim + row;
        x[idx] += alpha[n] * kp[idx] + zeta[n] * kt[idx];
    }
}

// Kernel: r = t - zeta * Akt
__global__ static void kernel_update_r(
    int batch,
    int dim,
    const double* __restrict__ t /* [batch * dim] */,
    const double* __restrict__ Akt /* [batch * dim] */,
    const double* __restrict__ zeta /* [batch] */,
    double* __restrict__ r /* out [batch * dim] */
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < dim && n < batch) {
        int idx = n * dim + row;
        r[idx] = t[idx] - zeta[n] * Akt[idx];
    }
}

extern "C" void bicgstab_cuda_wmma(
    int batch,
    int dim,
    double** A /* in [dim][dim] */,
    double** b /* in [batch][dim] */,
    double** x /* out [batch][dim] */,
    double tor,
    int max_steps
) {
    const size_t dim_dim_bytes = static_cast<size_t>(dim) * dim * sizeof(double);
    const size_t batch_dim_bytes = static_cast<size_t>(batch) * dim * sizeof(double);
    const size_t batch_bytes = static_cast<size_t>(batch) * sizeof(double);

    // Device buffers
    double *d_A, *d_b, *d_x;
    CUDA_CHECK(cudaMalloc(&d_A, dim_dim_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, batch_dim_bytes));
    CUDA_CHECK(cudaMalloc(&d_x, batch_dim_bytes));

    CUDA_CHECK(cudaMemcpy(d_A, A[0], dim_dim_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b[0], batch_dim_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, x[0], batch_dim_bytes, cudaMemcpyHostToDevice));

    // Work arrays
    double *d_p, *d_r, *d_r0, *d_t, *d_Ap, *d_Akp, *d_kt, *d_Akt, *d_kp;
    cudaMalloc(&d_p, batch_dim_bytes);
    cudaMalloc(&d_r, batch_dim_bytes);
    cudaMalloc(&d_r0, batch_dim_bytes);
    cudaMalloc(&d_t, batch_dim_bytes);
    cudaMalloc(&d_Ap, batch_dim_bytes);
    cudaMalloc(&d_Akp, batch_dim_bytes);
    cudaMalloc(&d_kt, batch_dim_bytes);
    cudaMalloc(&d_Akt, batch_dim_bytes);
    cudaMalloc(&d_kp, batch_dim_bytes);

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

    dim3 block2d(16, 16, 1);
    dim3 grid2d((dim + 15) / 16, (batch + 15) / 16, 1);
    int threads1d = 256;
    int blocks1d = (batch + threads1d - 1) / threads1d;

    // bnorm = sqrt(dot_product(dim, b, b));
    kernel_dot_product<<<blocks1d, threads1d>>>(batch, dim, d_b, d_b, d_bnorm);
    kernel_sqrt<<<blocks1d, threads1d>>>(batch, d_bnorm, d_bnorm);

    // r = b - A * x
    kernel_residual<<<grid2d,block2d>>>(batch, dim, d_A, d_x, d_b, d_r);
    cudaMemcpy(d_r0, d_r, batch_dim_bytes, cudaMemcpyDeviceToDevice);
    kernel_dot_product<<<blocks1d, threads1d>>>(batch, dim, d_r, d_r, d_rnorm);
    kernel_sqrt<<<blocks1d, threads1d>>>(batch, d_rnorm, d_rnorm);

    cudaMemset(d_p, 0, batch_dim_bytes);
    cudaMemset(d_alpha, 0, batch_bytes);
    cudaMemset(d_beta, 0, batch_bytes);
    cudaMemset(d_zeta, 0, batch_bytes);

    double* tmp = static_cast<double*>(malloc(sizeof(double) * batch)); // <-- Allocation: tmp

    // if (rnorm / bnorm < tor) { goto finalize; } // early exit
    kernel_div<<<blocks1d, threads1d>>>(batch, d_rnorm, d_bnorm, d_tmp);
    cudaMemcpy(tmp, d_tmp, batch_bytes, cudaMemcpyDeviceToHost);
    printf("Original relative residual norm [0] = %20.14e\n", tmp[0]);
    if (batch_lt(batch, tmp, tor)) { goto finalize; }

    // BiCGSTAB iteration 
    for (int step = 1; step <= max_steps; ++step) {
        // matvec(dim, A, p, Ap);
        kernel_matvec<<<grid2d, block2d>>>(batch, dim, d_A, d_p, d_Ap);

        // p[i] = r[i] + beta * (p[i] - zeta * Ap[i]);
        kernel_update_p<<<grid2d, block2d>>>(batch, dim, d_p, d_r, d_p, d_Ap, d_beta, d_zeta);
        cudaMemcpy(d_kp, d_p, batch_dim_bytes, cudaMemcpyDeviceToDevice);

        // matvec(dim, A, kp, Akp);
        // kernel_matvec<<<grid2d, block2d>>>(batch, dim, d_A, d_kp, d_Akp);
        launch_wmma_matvec(batch, dim, d_A, d_kp, d_Akp);

        // nom = dot_product(dim, r0, r);
        kernel_dot_product<<<blocks1d, threads1d>>>(batch, dim, d_r0, d_r, d_nom);

        // den = dot_product(dim, r0, Akp);
        kernel_dot_product<<<blocks1d, threads1d>>>(batch, dim, d_r0, d_Akp, d_den);

        // alpha = nom / den;
        kernel_div<<<blocks1d, threads1d>>>(batch, d_nom, d_den, d_alpha);

        // nom_old = nom;
        cudaMemcpy(d_nom_old, d_nom, batch_bytes, cudaMemcpyDeviceToDevice);

        // t[i] = r[i] - alpha * Akp[i];
        kernel_update_t<<<grid2d,block2d>>>(batch, dim, d_r, d_Akp, d_alpha, d_t);

        // kt[i] = t[i];
        cudaMemcpy(d_kt, d_t, batch_dim_bytes, cudaMemcpyDeviceToDevice);

        // matvec(dim, A, kt, Akt);
        kernel_matvec<<<grid2d,block2d>>>(batch, dim, d_A, d_kt, d_Akt);

        // nom = dot_product(dim, Akt, t);
        kernel_dot_product<<<blocks1d, threads1d>>>(batch, dim, d_Akt, d_t, d_nom);

        // den = dot_product(dim, Akt, Akt);
        kernel_dot_product<<<blocks1d, threads1d>>>(batch, dim, d_Akt, d_Akt, d_den);

        // zeta = nom / den;
        kernel_div<<<blocks1d, threads1d>>>(batch, d_nom, d_den, d_zeta);

        // x[i] = x[i] + alpha * kp[i] + zeta * kt[i];
        kernel_update_x<<<grid2d,block2d>>>(batch, dim, d_x, d_kp, d_kt, d_alpha, d_zeta);

        // r[i] = t[i] - zeta * Akt[i];
        kernel_update_r<<<grid2d,block2d>>>(batch, dim, d_t, d_Akt, d_zeta, d_r);

        // beta = alpha/zeta * dot(r0,r) / nom_old
        kernel_dot_product<<<blocks1d, threads1d>>>(batch, dim, d_r0, d_r, d_tmp); // dot_product(dim, r0, r)
        kernel_mul<<<blocks1d, threads1d>>>(batch, d_alpha, d_tmp, d_beta); // alpha * dot_product(dim, r0, r)
        kernel_div<<<blocks1d, threads1d>>>(batch, d_beta, d_zeta, d_beta); // alpha / zeta * dot_product(dim, r0, r)
        kernel_div<<<blocks1d, threads1d>>>(
            batch, d_beta, d_nom_old, d_beta); // alpha / zeta * dot_product(dim, r0, r) / nom_old

        // rnorm and check
        kernel_dot_product<<<blocks1d, threads1d>>>(batch, dim, d_r, d_r, d_rnorm); // dot_product(dim, r, r)
        kernel_sqrt<<<blocks1d, threads1d>>>(batch, d_rnorm, d_rnorm); // sqrt(dot_product(dim, r, r))

        // if (rnorm / bnorm < tor) { break; }
        kernel_div<<<blocks1d, threads1d>>>(batch, d_rnorm, d_bnorm, d_tmp);
        cudaMemcpy(tmp, d_tmp, batch_bytes, cudaMemcpyDeviceToHost);
        printf("  Step %d relative residual norm [0] = %20.14e\n", step, tmp[0]);
        if (batch_lt(batch, tmp, tor)) { break; }
    }

finalize:
    cudaMemcpy(x[0], d_x, batch_dim_bytes, cudaMemcpyDeviceToHost);

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
