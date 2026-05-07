#include <cstdio>
#include <cstdlib>
#include <x86intrin.h>

int main() {
  const int N = 16;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }

  const __m512 xj = _mm512_loadu_ps(x);
  const __m512 yj = _mm512_loadu_ps(y);
  const __m512 mj = _mm512_loadu_ps(m);
  const __m512 one = _mm512_set1_ps(1.0f);

  for(int i=0; i<N; i++) {
    const __mmask16 mask = 0xffff ^ (1 << i);
    const __m512 xi = _mm512_set1_ps(x[i]);
    const __m512 yi = _mm512_set1_ps(y[i]);
    const __m512 rx = _mm512_sub_ps(xi, xj);
    const __m512 ry = _mm512_sub_ps(yi, yj);
    const __m512 r2 = _mm512_add_ps(_mm512_mul_ps(rx, rx), _mm512_mul_ps(ry, ry));
    const __m512 safe_r2 = _mm512_mask_mov_ps(r2, 1 << i, one);
    const __m512 inv_r3 = _mm512_div_ps(one, _mm512_mul_ps(safe_r2, _mm512_sqrt_ps(safe_r2)));
    const __m512 fxi = _mm512_maskz_mul_ps(mask, _mm512_mul_ps(rx, mj), inv_r3);
    const __m512 fyi = _mm512_maskz_mul_ps(mask, _mm512_mul_ps(ry, mj), inv_r3);

    fx[i] = -_mm512_reduce_add_ps(fxi);
    fy[i] = -_mm512_reduce_add_ps(fyi);
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
