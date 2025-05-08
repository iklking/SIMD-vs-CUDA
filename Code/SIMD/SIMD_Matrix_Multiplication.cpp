// SIMD matrix multiplication using AVX2
#include <iostream>
#include <chrono>
#include <immintrin.h>
#include <malloc.h>
#include <cmath>

// Scalar matrix multiplication
void matmul_scalar(const float* A, const float* B, float* C, int N) {

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < N; ++k)
        sum += A[i * N + k] * B[k * N + j];
      C[i * N + j] = sum;
    }
  }

}

// SIMD matrix multiplication using AVX2
void matmul_simd(const float* A, const float* B, float* C, int N) {

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      __m256 sum_vec = _mm256_setzero_ps();
      int k = 0;
      for (; k <= N - 8; k += 8) {
         // Load 8 elements of row A and gather corresponding elements of column B
        __m256 a_vec = _mm256_loadu_ps(&A[i * N + k]);
        float b_temp[8];
        for (int t = 0; t < 8; ++t)
        b_temp[t] = B[(k + t) * N + j];
        __m256 b_vec = _mm256_loadu_ps(b_temp);
         // Multiply and accumulate
        sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
      }

      // Horizontal sum of SIMD register
      float sum_array[8];
      _mm256_storeu_ps(sum_array, sum_vec);
      float sum = 0.0f;
      for (int t = 0; t < 8; ++t)
          sum += sum_array[t];
  
      // Handle remaining elements
      for (; k < N; ++k)
        sum += A[i * N + k] * B[k * N + j];

      C[i * N + j] = sum;
    }
  }

}\

void run_test(int N) {

  float* A = (float*)_aligned_malloc(N * N * sizeof(float), 32);
  float* B = (float*)_aligned_malloc(N * N * sizeof(float), 32);
  float* C_scalar = (float*)_aligned_malloc(N * N * sizeof(float), 32);
  float* C_simd = (float*)_aligned_malloc(N * N * sizeof(float), 32);

  // Initialize matrices
  for (int i = 0; i < N * N; ++i) {
    A[i] = static_cast<float>(i % 100) * 0.5f;
    B[i] = static_cast<float>(i % 100) * 0.25f;
  }

  const int trials = 5;
  double scalar_total = 0;
  double simd_total = 0;
  // Run scalar
  for (int t = 0; t < trials; ++t) {
    auto start = std::chrono::high_resolution_clock::now();
    matmul_scalar(A, B, C_scalar, N);
    auto end = std::chrono::high_resolution_clock::now();
    scalar_total += std::chrono::duration<double, std::milli>(end - start).count();
  }

  // Run SIMD
  for (int t = 0; t < trials; ++t) {
    auto start = std::chrono::high_resolution_clock::now();
    matmul_simd(A, B, C_simd, N);
    auto end = std::chrono::high_resolution_clock::now();
    simd_total += std::chrono::duration<double, std::milli>(end - start).count();
  }

  // Verify correctness
  for (int i = 0; i < N * N; ++i) {
    if (fabs(C_scalar[i] - C_simd[i]) > 1e-2f) { // 1e-2 because floating point error increases
      std::cerr << "Mismatch at index " << i << ": " << C_scalar[i] << " vs " << C_simd[i] << "\n";
      break;
    }
  }

   // Calculate speedups
  double scalar_avg = scalar_total / trials;
  double simd_avg = simd_total / trials;

   // Print results
  std::cout << "N = " << N << "\tScalar Avg: " << scalar_avg << " ms" << "\tSIMD Avg: " << simd_avg << " ms" << "\tSpeedup: " << (scalar_avg / simd_avg) << "x\n";

   // Free matrices
  _aligned_free(A);
  _aligned_free(B);
  _aligned_free(C_scalar);
  _aligned_free(C_simd);
}

int main() {

  std::cout << "Running SIMD Matrix Multiplication tests...\n";
  for (int exp = 6; exp <= 10; ++exp) { // 2^6 = 64, 2^7 = 128, ..., 2^10 = 1024
    int N = 1 << exp;
    run_test(N);
  }
  std::cout << "Done.\n";
  return 0;
}
