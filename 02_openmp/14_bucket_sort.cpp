#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range,0);
#pragma omp parallel for
  for (int i=0; i<n; i++) {
#pragma omp atomic
    bucket[key[i]]++;
  }

  // Hillis-Steele parallel prefix sum (inclusive → exclusive)
  std::vector<int> offset(bucket), tmp(range);
  for (int step=1; step<range; step*=2) {
#pragma omp parallel for
    for (int i=0; i<range; i++)
      tmp[i] = offset[i] + (i>=step ? offset[i-step] : 0);
    std::swap(offset, tmp);
  }
  for (int i=range-1; i>0; i--) offset[i] = offset[i-1];
  offset[0] = 0;

#pragma omp parallel for
  for (int i=0; i<range; i++) {
    int j = offset[i];
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
