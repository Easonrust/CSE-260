#include "bl_config.h"
#include "bl_dgemm_kernel.h"
#include <immintrin.h>

#define a(i, j, ld) a[(i) * (ld) + (j)]
#define b(i, j, ld) b[(i) * (ld) + (j)]
#define c(i, j, ld) c[(i) * (ld) + (j)]

//
// C-based micorkernel
//
void bl_dgemm_ukr(int k,
                  int m,
                  int n,
                  const double *restrict a,
                  const double *restrict b,
                  double *c,
                  unsigned long long ldc,
                  aux_t *data)
{
    int l, j, i, len = m / 4 * 4;

    for (l = 0; l < k; ++l)
    {
        for (j = 0; j < n; ++j)
        {
            for (i = 0; i < len; i += 4)
            {
                // ldc is used here because a[] and b[] are not packed by the
                // starter code
                // cse260 - you can modify the leading indice to DGEMM_NR and DGEMM_MR as appropriate
                //
                c(i, j, ldc) += a(l, i, DGEMM_MR) * b(l, j, DGEMM_NR);
                c(i + 1, j, ldc) += a(l, i + 1, DGEMM_MR) * b(l, j, DGEMM_NR);
                c(i + 2, j, ldc) += a(l, i + 2, DGEMM_MR) * b(l, j, DGEMM_NR);
                c(i + 3, j, ldc) += a(l, i + 3, DGEMM_MR) * b(l, j, DGEMM_NR);
                //   printf("c[%d][%d] %lf * %lf = %lf \n", i, j, a( l, i, DGEMM_MR), b( l, j, DGEMM_NR ), a( l, i, DGEMM_MR) * b( l, j, DGEMM_NR ));
            }
            for (; i < m; ++i)
                c(i, j, ldc) += a(l, i, DGEMM_MR) * b(l, j, DGEMM_NR);
        }
    }
}

// cse260
// you can put your optimized kernels here
// - put the function prototypes in bl_dgemm_kernel.h
// - define BL_MICRO_KERNEL appropriately in bl_config.h
//

inline void bl_dgemm_avx_256_4_4(int k,
                                 int m,
                                 int n,
                                 const double *restrict a,
                                 const double *restrict b,
                                 double *c,
                                 unsigned long long ldc,
                                 aux_t *data)
{
    int l;
    register __m256d c00_c01_c02_c03 = _mm256_loadu_pd(c + 0 * ldc);
    register __m256d c10_c11_c12_c13 = _mm256_loadu_pd(c + 1 * ldc);
    register __m256d c20_c21_c22_c23 = _mm256_loadu_pd(c + 2 * ldc);
    register __m256d c30_c31_c32_c33 = _mm256_loadu_pd(c + 3 * ldc);

    for (l = 0; l < k; ++l)
    {
        register __m256d a0l_a0l_a0l_a0l = _mm256_broadcast_sd(a + l * 4 + 0);
        register __m256d a1l_a1l_a1l_a1l = _mm256_broadcast_sd(a + l * 4 + 1);
        register __m256d a2l_a2l_a2l_a2l = _mm256_broadcast_sd(a + l * 4 + 2);
        register __m256d a3l_a3l_a3l_a3l = _mm256_broadcast_sd(a + l * 4 + 3);

        register __m256d bl0_bl1_bl2_bl3 = _mm256_loadu_pd(b + l * 4);

        c00_c01_c02_c03 = _mm256_fmadd_pd(a0l_a0l_a0l_a0l, bl0_bl1_bl2_bl3, c00_c01_c02_c03);
        c10_c11_c12_c13 = _mm256_fmadd_pd(a1l_a1l_a1l_a1l, bl0_bl1_bl2_bl3, c10_c11_c12_c13);
        c20_c21_c22_c23 = _mm256_fmadd_pd(a2l_a2l_a2l_a2l, bl0_bl1_bl2_bl3, c20_c21_c22_c23);
        c30_c31_c32_c33 = _mm256_fmadd_pd(a3l_a3l_a3l_a3l, bl0_bl1_bl2_bl3, c30_c31_c32_c33);
    }
    _mm256_storeu_pd(c + 0 * ldc, c00_c01_c02_c03);
    _mm256_storeu_pd(c + 1 * ldc, c10_c11_c12_c13);
    _mm256_storeu_pd(c + 2 * ldc, c20_c21_c22_c23);
    _mm256_storeu_pd(c + 3 * ldc, c30_c31_c32_c33);
}

inline void bl_dgemm_avx_256_4_8(int k,
                                 int m,
                                 int n,
                                 const double *restrict a,
                                 const double *restrict b,
                                 double *c,
                                 unsigned long long ldc,
                                 aux_t *data)
{
    int l;
    register __m256d c00_c01_c02_c03 = _mm256_loadu_pd(c + 0 * ldc);
    register __m256d c04_c05_c06_c07 = _mm256_loadu_pd(c + 0 * ldc + 4);
    register __m256d c10_c11_c12_c13 = _mm256_loadu_pd(c + 1 * ldc);
    register __m256d c14_c15_c16_c17 = _mm256_loadu_pd(c + 1 * ldc + 4);
    register __m256d c20_c21_c22_c23 = _mm256_loadu_pd(c + 2 * ldc);
    register __m256d c24_c25_c26_c27 = _mm256_loadu_pd(c + 2 * ldc + 4);
    register __m256d c30_c31_c32_c33 = _mm256_loadu_pd(c + 3 * ldc);
    register __m256d c34_c35_c36_c37 = _mm256_loadu_pd(c + 3 * ldc + 4);

    for (l = 0; l < k; ++l)
    {
        register __m256d a0l_a0l_a0l_a0l = _mm256_broadcast_sd(a + l * 4 + 0);
        register __m256d a1l_a1l_a1l_a1l = _mm256_broadcast_sd(a + l * 4 + 1);
        register __m256d a2l_a2l_a2l_a2l = _mm256_broadcast_sd(a + l * 4 + 2);
        register __m256d a3l_a3l_a3l_a3l = _mm256_broadcast_sd(a + l * 4 + 3);

        register __m256d bl0_bl1_bl2_bl3 = _mm256_loadu_pd(b + l * 8);
        register __m256d bl4_bl5_bl6_bl7 = _mm256_loadu_pd(b + l * 8 + 4);

        c00_c01_c02_c03 = _mm256_fmadd_pd(a0l_a0l_a0l_a0l, bl0_bl1_bl2_bl3, c00_c01_c02_c03);
        c04_c05_c06_c07 = _mm256_fmadd_pd(a0l_a0l_a0l_a0l, bl4_bl5_bl6_bl7, c04_c05_c06_c07);
        c10_c11_c12_c13 = _mm256_fmadd_pd(a1l_a1l_a1l_a1l, bl0_bl1_bl2_bl3, c10_c11_c12_c13);
        c14_c15_c16_c17 = _mm256_fmadd_pd(a1l_a1l_a1l_a1l, bl4_bl5_bl6_bl7, c14_c15_c16_c17);
        c20_c21_c22_c23 = _mm256_fmadd_pd(a2l_a2l_a2l_a2l, bl0_bl1_bl2_bl3, c20_c21_c22_c23);
        c24_c25_c26_c27 = _mm256_fmadd_pd(a2l_a2l_a2l_a2l, bl4_bl5_bl6_bl7, c24_c25_c26_c27);
        c30_c31_c32_c33 = _mm256_fmadd_pd(a3l_a3l_a3l_a3l, bl0_bl1_bl2_bl3, c30_c31_c32_c33);
        c34_c35_c36_c37 = _mm256_fmadd_pd(a3l_a3l_a3l_a3l, bl4_bl5_bl6_bl7, c34_c35_c36_c37);
    }
    _mm256_storeu_pd(c + 0 * ldc, c00_c01_c02_c03);
    _mm256_storeu_pd(c + 0 * ldc + 4, c04_c05_c06_c07);
    _mm256_storeu_pd(c + 1 * ldc, c10_c11_c12_c13);
    _mm256_storeu_pd(c + 1 * ldc + 4, c14_c15_c16_c17);
    _mm256_storeu_pd(c + 2 * ldc, c20_c21_c22_c23);
    _mm256_storeu_pd(c + 2 * ldc + 4, c24_c25_c26_c27);
    _mm256_storeu_pd(c + 3 * ldc, c30_c31_c32_c33);
    _mm256_storeu_pd(c + 3 * ldc + 4, c34_c35_c36_c37);
}
