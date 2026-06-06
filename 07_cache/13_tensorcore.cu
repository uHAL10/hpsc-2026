#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#include <mma.h>
#include <chrono>
using namespace std;
using namespace nvcuda;

__global__ void convert_to_half(int64_t size, const float *src, half *dst) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size) dst[i] = __float2half(src[i]);
}

__global__ void kernel(int dim_m, int dim_n, int dim_k,
		       half *d_a, half *d_b, float *d_c) {
  constexpr int tile_m = 128;
  constexpr int tile_n = 128;
  constexpr int tile_k = 128;
  constexpr int skew = 8;
  constexpr int stride_m = tile_m + skew;
  constexpr int stride_n = tile_n + skew;
  int offset_a_m = tile_m * blockIdx.x;
  int offset_b_n = tile_n * blockIdx.y;
  int warp_id = threadIdx.x / 32;
  int warp_m = warp_id & 3;
  int warp_n = warp_id >> 2;

  __shared__ half block_a[tile_k][stride_m];
  __shared__ half block_b[tile_k][stride_n];

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];
  #pragma unroll
  for (int r = 0; r < 2; r++)
    #pragma unroll
    for (int c = 0; c < 4; c++)
      wmma::fill_fragment(acc[r][c], 0.0f);

  for (int k = 0; k < dim_k; k += tile_k) {
    __syncthreads();
    #pragma unroll
    for (int idx = threadIdx.x; idx < tile_k * tile_m; idx += 256) {
      int row = idx / tile_m;
      int col = idx - row * tile_m;
      block_a[row][col] = d_a[(k + row) * dim_m + offset_a_m + col];
    }
    #pragma unroll
    for (int idx = threadIdx.x; idx < tile_k * tile_n; idx += 256) {
      int col = idx / tile_k;
      int row = idx - col * tile_k;
      block_b[row][col] = d_b[(offset_b_n + col) * dim_k + k + row];
    }
    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < tile_k; kk += 16) {
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[4];
      #pragma unroll
      for (int c = 0; c < 4; c++) {
        wmma::load_matrix_sync(b_frag[c], &block_b[kk][(warp_n * 4 + c) * 16], stride_n);
      }

      #pragma unroll
      for (int r = 0; r < 2; r++) {
        int row_tile = warp_m * 2 + r;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
        wmma::load_matrix_sync(a_frag, &block_a[kk][row_tile * 16], stride_m);
        #pragma unroll
        for (int c = 0; c < 4; c++) {
          wmma::mma_sync(acc[r][c], a_frag, b_frag[c], acc[r][c]);
        }
      }
    }
  }
  #pragma unroll
  for (int r = 0; r < 2; r++) {
    #pragma unroll
    for (int c = 0; c < 4; c++) {
      int c_m = offset_a_m + (warp_m * 2 + r) * 16;
      int c_n = offset_b_n + (warp_n * 4 + c) * 16;
      if (c_n < dim_n && c_m < dim_m)
        wmma::store_matrix_sync(&d_c[c_n * dim_m + c_m], acc[r][c], dim_m, wmma::mem_col_major);
    }
  }
}

int main(int argc, const char **argv) {
  int m = 10240;
  int k = 4096;
  int n = 8192;
  float alpha = 1.0;
  float beta = 0.0;
  int Nt = 10;
  float *A, *B, *C, *C2;
  half *Ah, *Bh;
  cudaMallocManaged(&A, m * k * sizeof(float));
  cudaMallocManaged(&B, k * n * sizeof(float));
  cudaMallocManaged(&C, m * n * sizeof(float));
  cudaMallocManaged(&C2, m * n * sizeof(float));
  cudaMallocManaged(&Ah, m * k * sizeof(half));
  cudaMallocManaged(&Bh, k * n * sizeof(half));
  for (int i=0; i<m; i++)
    for (int j=0; j<k; j++)
      A[k*i+j] = drand48();
  for (int i=0; i<k; i++)
    for (int j=0; j<n; j++)
      B[n*i+j] = drand48();
  for (int i=0; i<n; i++)
    for (int j=0; j<m; j++)
      C[m*i+j] = C2[m*i+j] = 0;
  int conv_block = 256;
  convert_to_half<<<(int64_t(m) * k + conv_block - 1) / conv_block, conv_block>>>(int64_t(m) * k, A, Ah);
  convert_to_half<<<(int64_t(k) * n + conv_block - 1) / conv_block, conv_block>>>(int64_t(k) * n, B, Bh);
  cudaDeviceSynchronize();
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  auto tic = chrono::steady_clock::now();
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    cublasGemmEx(cublas_handle,
		 CUBLAS_OP_N,
		 CUBLAS_OP_N,
		 m,
		 n,
		 k,
		 &alpha,
		 A, CUDA_R_32F, m,
		 B, CUDA_R_32F, k,
		 &beta,
		 C, CUDA_R_32F, m,
		 CUBLAS_COMPUTE_32F_FAST_16F,
		 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();
  }
  auto toc = chrono::steady_clock::now();
  int64_t num_flops = (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
  double tcublas = chrono::duration<double>(toc - tic).count() / Nt;
  double cublas_flops = double(num_flops) / tcublas / 1.0e9;
  int tile_m = 128;
  int tile_n = 128;
  dim3 block = dim3(256);
  dim3 grid = dim3((m+tile_m-1)/tile_m, (n+tile_n-1)/tile_n);
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    kernel<<< grid, block >>>(m,
			      n,
			      k,
			      Ah,
			      Bh,
			      C2);
    cudaDeviceSynchronize();
  }
  toc = chrono::steady_clock::now();
  double tcutlass = chrono::duration<double>(toc - tic).count() / Nt;
  double cutlass_flops = double(num_flops) / tcutlass / 1.0e9;
  printf("CUBLAS: %.2f Gflops, CUTLASS: %.2f Gflops\n", cublas_flops, cutlass_flops);
  double err = 0;
  for (int i=0; i<n; i++) {
    for (int j=0; j<m; j++) {
      err += fabs(C[m*i+j] - C2[m*i+j]);
    }
  }
  printf("error: %lf\n", err/n/m);
  cudaFree(A);
  cudaFree(B);
  cudaFree(Ah);
  cudaFree(Bh);
  cudaFree(C);
  cudaFree(C2);
  cublasDestroy(cublas_handle);
}
