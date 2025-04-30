#include <mma.h>
#include <iostream>
#include <cuda_runtime.h>

using namespace nvcuda;

__global__ void wmma_double_kernel(const double* a, const double* b, double* c) {
    // WMMA fragments
    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 8, 8, 4, double> c_frag;

    // Load from global memory
    wmma::load_matrix_sync(a_frag, a, 4); // leading dimension = 4 (A is 8x4)
    wmma::load_matrix_sync(b_frag, b, 4); // leading dimension = 4 (B is 4x8)
    wmma::fill_fragment(c_frag, 0.0);

    // Matrix multiply-accumulate
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store the result
    wmma::store_matrix_sync(c, c_frag, 8, wmma::mem_row_major); // C is 8x8
}

int main() {
    const int M = 8, N = 8, K = 4;
    const int A_elems = M * K;
    const int B_elems = K * N;
    const int C_elems = M * N;

    double h_a[A_elems], h_b[B_elems], h_c[C_elems];

    // A: row-major 8x4 (leading dim = 4)
    // B: col-major 4x8 (leading dim = 4)
    // A[i,j] = 1.0, B[i,j] = 1.0 → C = A × B = 4.0 になる
    for (int i = 0; i < A_elems; ++i) h_a[i] = 1.0;
    for (int i = 0; i < B_elems; ++i) h_b[i] = 1.0;

    double *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, sizeof(h_a));
    cudaMalloc(&d_b, sizeof(h_b));
    cudaMalloc(&d_c, sizeof(h_c));
    cudaMemcpy(d_a, h_a, sizeof(h_a), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(h_b), cudaMemcpyHostToDevice);

    // Launch one warp (32 threads)
    wmma_double_kernel<<<1, 32>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, sizeof(h_c), cudaMemcpyDeviceToHost);

    std::cout << "C matrix:\n";
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << h_c[i * N + j] << " ";
        }
        std::cout << "\n";
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
