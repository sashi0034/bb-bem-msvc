#include <mma.h>
#include <cuda_fp16.h>
#include <iostream>

using namespace nvcuda;

__global__ void wmma_gemm_16x16(half* a, half* b, float* c) {
    // 16x16x16 の WMMA fragment を定義
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // 初期化
    wmma::fill_fragment(c_frag, 0.0f);

    // グローバルメモリから WMMA fragment にロード
    wmma::load_matrix_sync(a_frag, a, 16);
    wmma::load_matrix_sync(b_frag, b, 16);

    // 行列乗算 & 累積
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // 結果をグローバルメモリに保存
    wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_col_major);
}

int main() {
    constexpr int M = 16, N = 16, K = 16;

    half a[M * K], b[K * N];
    float c[M * N];

    // 行列 a, b を 1.0 で初期化（任意）
    for (int i = 0; i < M * K; i++) a[i] = __float2half(1.0f);
    for (int i = 0; i < K * N; i++) b[i] = __float2half(1.0f);

    // GPUメモリ確保
    half *d_a, *d_b;
    float* d_c;
    cudaMalloc(&d_a, sizeof(a));
    cudaMalloc(&d_b, sizeof(b));
    cudaMalloc(&d_c, sizeof(c));

    // 転送
    cudaMemcpy(d_a, a, sizeof(a), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(b), cudaMemcpyHostToDevice);

    // カーネル実行
    wmma_gemm_16x16<<<1, 32>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();

    // 結果取得
    cudaMemcpy(c, d_c, sizeof(c), cudaMemcpyDeviceToHost);

    // 表示（結果はすべて16.0のはず）
    std::cout << "Result (first row):\n";
    for (int i = 0; i < N; ++i) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    // 後始末
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
