#include <cstdio>
#include <cstdlib>

__global__ void init_bucket(int *bucket, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < range)
    bucket[i] = 0;
}

__global__ void count_bucket(int *key, int *bucket, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    atomicAdd(&bucket[key[i]], 1);
}

__global__ void write_bucket(int *key, int *bucket, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < range) {
    int offset = 0;
    for (int j=0; j<i; j++)
      offset += bucket[j];
    for (int j=0; j<bucket[i]; j++)
      key[offset+j] = i;
  }
}

int main() {
  const int n = 50;
  const int range = 5;
  int *key, *bucket;
  cudaMallocManaged(&key, n*sizeof(int));
  cudaMallocManaged(&bucket, range*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  init_bucket<<<1,range>>>(bucket, range);
  cudaDeviceSynchronize();
  count_bucket<<<(n+255)/256,256>>>(key, bucket, n);
  cudaDeviceSynchronize();
  write_bucket<<<1,range>>>(key, bucket, range);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");

  cudaFree(key);
  cudaFree(bucket);
}
