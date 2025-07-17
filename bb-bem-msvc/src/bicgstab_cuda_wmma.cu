#include <mma.h>
using namespace nvcuda;

#include <stdio.h>

#include "bicgstab_cuda_wmma.h"

#ifndef NO_TRACE
#define TRACE(...) printf(__VA_ARGS__)
#else
#define TRACE(...) do {} while (0)
#endif

#define CUDA_CHECK(err) do { \
    cudaError_t _e = (err); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA Error %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_e)); \
        return; \
    } \
} while (0)

static constexpr int WMMA_M = 8;
static constexpr int WMMA_N = 8;
static constexpr int WMMA_K = 4;

__host__ __device__ int tcl_at(int row, int col, int num_cols) {
    static_assert(WMMA_M == 8 && WMMA_N == 8, "Must be 8 for this implementation");

    const int tiles_per_row = num_cols >> 3; // num_cols / 8

    const int coarse_row = row >> 3; // row / 8
    const int coarse_col = col >> 3; // col / 8
    const int fine_row = row & 7; // row % 8
    const int fine_col = col & 7; // col % 8

    // (coarse_row * tiles_per_row + coarse_col) * 64 + (fine_row * 8 + fine_col)
    return ((coarse_row * tiles_per_row + coarse_col) << 6) + (fine_row << 3) + fine_col;
}

// __global__ static void kernel_matvec(
//     int batch,
//     int dim,
//     const double* __restrict__ mat /* [dim * dim] */,
//     const double* __restrict__ P /* [dim * batch] */,
//     double* __restrict__ Q /* out [dim * batch] */
// ) {
//     const int row = blockIdx.x * blockDim.x + threadIdx.x;
//     const int n = blockIdx.y * blockDim.y + threadIdx.y;
//     if (row < dim && n < batch) {
//         double sum = 0.0;
//         for (int col = 0; col < dim; ++col) {
//             sum += mat[tcl_at(row, col, dim)] * P[tcl_at(col, n, batch)];;
//         }
//
//         Q[tcl_at(row, n, batch)] = sum;
//     }
// }

constexpr int WMMA_WARPS = 2;

// Kernel: Q[row, n] = sum_col A[row, col] * P[col, n]
__global__ void wmma_matvec(
    int batch, /* 8-aligned */
    int dim, /* 8-aligned */
    const double* __restrict__ A /* [dim * dim] */,
    const double* __restrict__ P /* [dim * batch] */,
    double* __restrict__ Q /* out [dim * batch] */
) {
    constexpr int c_tile_elems = WMMA_M * WMMA_N; // 8x8 tiles

    __shared__ double smem[WMMA_WARPS * c_tile_elems]; // 2 warps, 8x8 tiles

    const int warpId = threadIdx.x >> 5; // threadIdx.x / 32

    // index of this tile
    const int coarse_row = blockIdx.x;
    const int coarse_col = blockIdx.y;

    // const int block_row = blockIdx.x * WMMA_M;
    // const int block_col = blockIdx.y * WMMA_N;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, double> c_frag;
    wmma::fill_fragment(c_frag, 0.0);

    const int num_k_tiles = dim / WMMA_K;
    const int num_k_tiles_per_warp = num_k_tiles / WMMA_WARPS; // 4 tiles per warp
    for (int k_tile = warpId * num_k_tiles_per_warp; k_tile < (warpId + 1) * num_k_tiles_per_warp; ++k_tile) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, double, wmma::row_major> a_frag; // 8x4
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, double, wmma::row_major> b_frag; // 4x8

        const double* a_tile_ptr = &A[tcl_at(coarse_row * WMMA_M, k_tile * WMMA_K, dim)];
        const double* b_tile_ptr = &P[tcl_at(k_tile * WMMA_K, coarse_col * WMMA_N, batch)];

        wmma::load_matrix_sync(a_frag, a_tile_ptr, 8);
        wmma::load_matrix_sync(b_frag, b_tile_ptr, 8);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(&smem[warpId * c_tile_elems], c_frag, 8, wmma::mem_row_major);

    __syncthreads();

    //  const int warpId = threadIdx.x >> 5; // threadIdx.x / 32
    if (warpId == 0 && threadIdx.x == 0) {
        // TODO: ここの高速化
        double* c_tile_ptr = &Q[tcl_at(coarse_row * WMMA_M, coarse_col * WMMA_N, batch)];
        for (int i = 0; i < c_tile_elems; ++i) {
            c_tile_ptr[i] = smem[i];
        }

        for (int w = 1; w < WMMA_WARPS; ++w) {
            for (int i = 0; i < c_tile_elems; ++i) {
                c_tile_ptr[i] += smem[w * c_tile_elems + i];
            }
        }
    }
}

static void launch_wmma_matvec(
    int batch, /* 8-aligned */
    int dim, /* 8-aligned */
    const double* d_A /* [dim * dim] */,
    const double* d_P /* [dim * batch] */,
    double* d_Q /* out [dim * batch] */
) {
    const dim3 grid(dim / WMMA_M, batch / WMMA_N);
    const dim3 block(32 * WMMA_WARPS, 1, 1); // one warp per tile

    wmma_matvec<<<grid, block>>>(batch, dim, d_A, d_P, d_Q);
}

// Kernel: R = B - A * X
__global__ static void kernel_residual(
    int batch,
    int dim,
    const double* __restrict__ A /* [dim * dim] */,
    const double* __restrict__ X /* [dim * batch] */,
    const double* __restrict__ B /* [dim * batch] */,
    double* __restrict__ R /* out [dim * batch] */
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < dim && n < batch) {
        double val = B[tcl_at(row, n, batch)];
        for (int col = 0; col < dim; ++col) {
            val -= A[tcl_at(row, col, dim)] * X[tcl_at(col, n, batch)];
        }

        R[tcl_at(row, n, batch)] = val;
    }
}

// Kernel: out[n] = dot( X[:,n], Y[:,n] )
__global__ static void kernel_dot_product(
    int batch,
    int dim,
    const double* __restrict__ X /* [dim * batch] */,
    const double* __restrict__ Y /* [dim * batch] */,
    double* __restrict__ out /* out [batch] */
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < batch) {
        double sum = 0.0;
        for (int i = 0; i < dim; ++i) {
            sum += X[tcl_at(i, n, batch)] * Y[tcl_at(i, n, batch)];
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
    double* __restrict__ out /* out [dim * batch] */,
    const double* __restrict__ r /* [dim * batch] */,
    const double* __restrict__ p /* [dim * batch] */,
    const double* __restrict__ Ap /* [dim * batch] */,
    const double* __restrict__ beta /* [batch] */,
    const double* __restrict__ zeta /* [batch] */
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < dim && n < batch) {
        const int idx = tcl_at(row, n, batch);
        out[tcl_at(row, n, batch)] = r[idx] + beta[n] * (p[idx] - zeta[n] * Ap[idx]);
    }
}

// Kernel: t = r - alpha * Akp
__global__ static void kernel_update_t(
    int batch,
    int dim,
    const double* __restrict__ r /* [dim * batch] */,
    const double* __restrict__ Akp /* [dim * batch] */,
    const double* __restrict__ alpha /* [batch] */,
    double* __restrict__ t /* out [dim * batch] */
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < dim && n < batch) {
        const int idx = tcl_at(row, n, batch);
        t[idx] = r[idx] - alpha[n] * Akp[idx];
    }
}

// Kernel: x += alpha * kp + zeta * kt
__global__ static void kernel_update_x(
    int batch,
    int dim,
    double* __restrict__ x /* inout [dim * batch] */,
    const double* __restrict__ kp /* [dim * batch] */,
    const double* __restrict__ kt /* [dim * batch] */,
    const double* __restrict__ alpha /* [batch] */,
    const double* __restrict__ zeta /* [batch] */
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < dim && n < batch) {
        const int idx = tcl_at(row, n, batch);
        x[idx] += alpha[n] * kp[idx] + zeta[n] * kt[idx];
    }
}

// Kernel: r = t - zeta * Akt
__global__ static void kernel_update_r(
    int batch,
    int dim,
    const double* __restrict__ t /* [dim * batch] */,
    const double* __restrict__ Akt /* [dim * batch] */,
    const double* __restrict__ zeta /* [batch] */,
    double* __restrict__ r /* out [dim * batch] */
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < dim && n < batch) {
        const int idx = tcl_at(row, n, batch);
        r[idx] = t[idx] - zeta[n] * Akt[idx];
    }
}

extern "C" void bicgstab_cuda_wmma(
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
    kernel_dot_product<<<blocks1d, threads1d>>>(batch, dim, d_b, d_b, d_bnorm);
    kernel_sqrt<<<blocks1d, threads1d>>>(batch, d_bnorm, d_bnorm);

    // r = b - A * x
    kernel_residual<<<grid2d,block2d>>>(batch, dim, d_A, d_x, d_b, d_r);
    cudaMemcpy(d_r0, d_r, dim_batch_bytes, cudaMemcpyDeviceToDevice);
    kernel_dot_product<<<blocks1d, threads1d>>>(batch, dim, d_r, d_r, d_rnorm);
    kernel_sqrt<<<blocks1d, threads1d>>>(batch, d_rnorm, d_rnorm);

    cudaMemset(d_p, 0, dim_batch_bytes);
    cudaMemset(d_alpha, 0, batch_bytes);
    cudaMemset(d_beta, 0, batch_bytes);
    cudaMemset(d_zeta, 0, batch_bytes);

    double* tmp = static_cast<double*>(malloc(sizeof(double) * batch)); // <-- Allocation: tmp

    // if (rnorm / bnorm < tor) { goto finalize; } // early exit
    kernel_div<<<blocks1d, threads1d>>>(batch, d_rnorm, d_bnorm, d_tmp);
    cudaMemcpy(tmp, d_tmp, batch_bytes, cudaMemcpyDeviceToHost);
    TRACE("Original relative residual norm [0] = %20.14e\n", tmp[0]);
    if (batch_lt(batch, tmp, tor)) { goto finalize; }

    // BiCGSTAB iteration 
    for (int step = 1; step <= max_steps; ++step) {
        // matvec(dim, A, p, Ap);
        // kernel_matvec<<<grid2d, block2d>>>(batch, dim, d_A, d_p, d_Ap);
        launch_wmma_matvec(batch, dim, d_A, d_p, d_Ap);

        // p[i] = r[i] + beta * (p[i] - zeta * Ap[i]);
        kernel_update_p<<<grid2d, block2d>>>(batch, dim, d_p, d_r, d_p, d_Ap, d_beta, d_zeta);
        cudaMemcpy(d_kp, d_p, dim_batch_bytes, cudaMemcpyDeviceToDevice);

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
        cudaMemcpy(d_kt, d_t, dim_batch_bytes, cudaMemcpyDeviceToDevice);

        //  matvec(dim, A, kt, Akt);
        // kernel_matvec<<<grid2d,block2d>>>(batch, dim, d_A, d_kt, d_Akt);
        launch_wmma_matvec(batch, dim, d_A, d_kt, d_Akt);

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
        TRACE("  Step %d relative residual norm [0] = %20.14e\n", step, tmp[0]);
        if (batch_lt(batch, tmp, tor)) { break; }
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

extern "C" int tensorcore_layout_at(int row, int col, int num_cols) {
    return tcl_at(row, col, num_cols);
}
