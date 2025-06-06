#include <mma.h>
// using namespace nvcuda;

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

// Kernel: q[row] = sum_col A[row, col] * p[col]
__global__ static void kernel_matvec(
    int dim,
    const double* __restrict__ A, // [dim * dim]
    const double* __restrict__ p, // [dim]
    double* __restrict__ q // out [dim]
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < dim) {
        double sum = 0.0;
        const double* Arow = A + row * dim;
        for (int col = 0; col < dim; ++col) {
            sum += Arow[col] * p[col];
        }
        q[row] = sum;
    }
}

// Kernel: r[row] = b[row] - sum_col A[row, col] * x[col]
__global__ static void kernel_residual(
    int dim,
    const double* __restrict__ A, // [dim * dim]
    const double* __restrict__ x, // [dim]
    const double* __restrict__ b, // [dim]
    double* __restrict__ r // out [dim]
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < dim) {
        double val = b[row];
        const double* Arow = A + row * dim;
        for (int col = 0; col < dim; ++col) {
            val -= Arow[col] * x[col];
        }
        r[row] = val;
    }
}

// Kernel: partial-block reduction for dot product; accumulates into out[0] via atomicAdd
__global__ static void kernel_dot(
    int dim,
    const double* __restrict__ x, // [dim]
    const double* __restrict__ y, // [dim]
    double* __restrict__ out // out[1]
) {
    __shared__ double sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    double sum = 0.0;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < dim; i += stride) {
        sum += x[i] * y[i];
    }
    sdata[tid] = sum;

    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

// Kernel: p[row] = r[row] + beta * (p[row] - zeta * Ap[row])
__global__ static void kernel_update_p(
    int dim,
    double* __restrict__ p, // inout [dim]
    const double* __restrict__ r, // [dim]
    const double* __restrict__ Ap, // [dim]
    double beta,
    double zeta
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < dim) {
        double old_p = p[row];
        p[row] = r[row] + beta * (old_p - zeta * Ap[row]);
    }
}

// Kernel: t[row] = r[row] - alpha * Akp[row]
__global__ static void kernel_update_t(
    int dim,
    const double* __restrict__ r, // [dim]
    const double* __restrict__ Akp, // [dim]
    double alpha,
    double* __restrict__ t // out [dim]
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < dim) {
        t[row] = r[row] - alpha * Akp[row];
    }
}

// Kernel: x[row] += alpha * kp[row] + zeta * kt[row]
__global__ static void kernel_update_x(
    int dim,
    double* __restrict__ x, // inout [dim]
    const double* __restrict__ kp, // [dim]
    const double* __restrict__ kt, // [dim]
    double alpha,
    double zeta
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < dim) {
        x[row] += alpha * kp[row] + zeta * kt[row];
    }
}

// Kernel: r[row] = t[row] - zeta * Akt[row]
__global__ static void kernel_update_r(
    int dim,
    const double* __restrict__ t, // [dim]
    const double* __restrict__ Akt, // [dim]
    double zeta,
    double* __restrict__ r // out [dim]
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < dim) {
        r[row] = t[row] - zeta * Akt[row];
    }
}

extern "C" void bicgstab_cuda(
    int dim,
    double** A, // in [dim][dim]
    double* b, // in [dim]
    double* x, // out [dim]
    double tor,
    int max_steps
) {
    const size_t dim_dim_bytes = static_cast<size_t>(dim) * dim * sizeof(double);
    const size_t vec_bytes = static_cast<size_t>(dim) * sizeof(double);

    // Device buffers
    double *d_A, *d_b, *d_x;
    CUDA_CHECK(cudaMalloc(&d_A, dim_dim_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_x, vec_bytes));

    CUDA_CHECK(cudaMemcpy(d_A, A[0], dim_dim_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, vec_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, x, vec_bytes, cudaMemcpyHostToDevice));

    // Work arrays
    double *d_p, *d_r, *d_r0, *d_t, *d_Ap, *d_Akp, *d_kt, *d_Akt, *d_kp;
    CUDA_CHECK(cudaMalloc(&d_p, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_r, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_r0, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_t, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_Ap, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_Akp, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_kt, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_Akt, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_kp, vec_bytes));

    // Temporary scalar for dot products
    double* d_dot;
    CUDA_CHECK(cudaMalloc(&d_dot, sizeof(double)));

    // Host scalars
    double host_bnorm = 0.0, host_rnorm = 0.0;
    double host_nom = 0.0, host_nom_old = 0.0, host_den = 0.0;
    double alpha = 0.0, beta = 0.0, zeta = 0.0;
    double temp_dot = 0.0;

    // Launch configuration
    int threads = 256;
    int blocks = (dim + threads - 1) / threads;

    // bnorm = sqrt(dot(b, b))
    CUDA_CHECK(cudaMemset(d_dot, 0, sizeof(double)));
    kernel_dot<<<blocks, threads>>>(dim, d_b, d_b, d_dot);
    CUDA_CHECK(cudaMemcpy(&host_bnorm, d_dot, sizeof(double), cudaMemcpyDeviceToHost));
    host_bnorm = sqrt(host_bnorm);

    // r = b - A * x
    kernel_residual<<<blocks, threads>>>(dim, d_A, d_x, d_b, d_r);

    // r0 = r
    CUDA_CHECK(cudaMemcpy(d_r0, d_r, vec_bytes, cudaMemcpyDeviceToDevice));

    // rnorm = sqrt(dot(r, r))
    CUDA_CHECK(cudaMemset(d_dot, 0, sizeof(double)));
    kernel_dot<<<blocks, threads>>>(dim, d_r, d_r, d_dot);
    CUDA_CHECK(cudaMemcpy(&host_rnorm, d_dot, sizeof(double), cudaMemcpyDeviceToHost));
    host_rnorm = sqrt(host_rnorm);

    // Initialize p, alpha, beta, zeta
    CUDA_CHECK(cudaMemset(d_p, 0, vec_bytes));
    alpha = beta = zeta = 0.0;

    printf("Original relative residual norm = %20.14e\n", host_rnorm / host_bnorm);

    if (host_rnorm / host_bnorm < tor) {
        goto finalize;
    }

    // BiCGSTAB iteration
    for (int step = 1; step <= max_steps; ++step) {
        // Ap = A * p
        kernel_matvec<<<blocks, threads>>>(dim, d_A, d_p, d_Ap);

        // p = r + beta * (p - zeta * Ap)
        kernel_update_p<<<blocks, threads>>>(dim, d_p, d_r, d_Ap, beta, zeta);

        // kp = p
        CUDA_CHECK(cudaMemcpy(d_kp, d_p, vec_bytes, cudaMemcpyDeviceToDevice));

        // Akp = A * kp
        kernel_matvec<<<blocks, threads>>>(dim, d_A, d_kp, d_Akp);

        // nom = dot(r0, r)
        CUDA_CHECK(cudaMemset(d_dot, 0, sizeof(double)));
        kernel_dot<<<blocks, threads>>>(dim, d_r0, d_r, d_dot);
        CUDA_CHECK(cudaMemcpy(&host_nom, d_dot, sizeof(double), cudaMemcpyDeviceToHost));

        // den = dot(r0, Akp)
        CUDA_CHECK(cudaMemset(d_dot, 0, sizeof(double)));
        kernel_dot<<<blocks, threads>>>(dim, d_r0, d_Akp, d_dot);
        CUDA_CHECK(cudaMemcpy(&host_den, d_dot, sizeof(double), cudaMemcpyDeviceToHost));

        alpha = host_nom / host_den;
        host_nom_old = host_nom;

        // t = r - alpha * Akp
        kernel_update_t<<<blocks, threads>>>(dim, d_r, d_Akp, alpha, d_t);

        // kt = t
        CUDA_CHECK(cudaMemcpy(d_kt, d_t, vec_bytes, cudaMemcpyDeviceToDevice));

        // Akt = A * kt
        kernel_matvec<<<blocks, threads>>>(dim, d_A, d_kt, d_Akt);

        // nom = dot(Akt, t)
        CUDA_CHECK(cudaMemset(d_dot, 0, sizeof(double)));
        kernel_dot<<<blocks, threads>>>(dim, d_Akt, d_t, d_dot);
        CUDA_CHECK(cudaMemcpy(&host_nom, d_dot, sizeof(double), cudaMemcpyDeviceToHost));

        // den = dot(Akt, Akt)
        CUDA_CHECK(cudaMemset(d_dot, 0, sizeof(double)));
        kernel_dot<<<blocks, threads>>>(dim, d_Akt, d_Akt, d_dot);
        CUDA_CHECK(cudaMemcpy(&host_den, d_dot, sizeof(double), cudaMemcpyDeviceToHost));

        zeta = host_nom / host_den;

        // x = x + alpha * kp + zeta * kt
        kernel_update_x<<<blocks, threads>>>(dim, d_x, d_kp, d_kt, alpha, zeta);

        // r = t - zeta * Akt
        kernel_update_r<<<blocks, threads>>>(dim, d_t, d_Akt, zeta, d_r);

        // Compute beta = (alpha / zeta) * (dot(r0, r) / host_nom_old)
        CUDA_CHECK(cudaMemset(d_dot, 0, sizeof(double)));
        kernel_dot<<<blocks, threads>>>(dim, d_r0, d_r, d_dot);
        CUDA_CHECK(cudaMemcpy(&temp_dot, d_dot, sizeof(double), cudaMemcpyDeviceToHost));
        beta = (alpha / zeta) * (temp_dot / host_nom_old);

        // rnorm = sqrt(dot(r, r))
        CUDA_CHECK(cudaMemset(d_dot, 0, sizeof(double)));
        kernel_dot<<<blocks, threads>>>(dim, d_r, d_r, d_dot);
        CUDA_CHECK(cudaMemcpy(&host_rnorm, d_dot, sizeof(double), cudaMemcpyDeviceToHost));
        host_rnorm = sqrt(host_rnorm);

        printf("  Step %d relative residual norm = %20.14e\n", step, host_rnorm / host_bnorm);

        if (host_rnorm / host_bnorm < tor) {
            break;
        }
    }

finalize:
    // Copy solution back to host
    CUDA_CHECK(cudaMemcpy(x, d_x, vec_bytes, cudaMemcpyDeviceToHost));

    // Free device memory
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
    cudaFree(d_dot);
}
