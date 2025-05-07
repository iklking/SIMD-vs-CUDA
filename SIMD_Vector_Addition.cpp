// SIMD vector addition using AVX2
#include <iostream>
#include <immintrin.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
// Perform scalar vector addition
void vector_add_scalar(const float* A, const float* B, float* C, int N) {

  for (int i = 0; i < N; ++i)
    C[i] = A[i] + B[i];
  }

  // Perform SIMD addition with AVX2
  void vector_add_avx2(const float* A, const float* B, float* C, int N) {
    int i = 0;
    for (; i <= N - 8; i += 8) {
      // Load 8 floats from each array
      __m256 a = _mm256_loadu_ps(&A[i]);
      __m256 b = _mm256_loadu_ps(&B[i]);
  
      // Add the vectors
      __m256 c = _mm256_add_ps(a, b);
  
      // Store the result
      _mm256_storeu_ps(&C[i], c);
    }
     // Handle remaining elements
    for (; i < N; ++i) {
      C[i] = A[i] + B[i];
    }
}
// Run tests across different input sizes and measure performance
void run_test(int N) {

  float* A = (float*)_aligned_malloc(N * sizeof(float), 32);
  float* B = (float*)_aligned_malloc(N * sizeof(float), 32);
  float* C_scalar = (float*)_aligned_malloc(N * sizeof(float), 32);
  float* C_simd = (float*)_aligned_malloc(N * sizeof(float), 32);

   // Initialize input arrays
  for (int i = 0; i < N; ++i) {
    A[i] = i;
    B[i] = i * 0.5f;
  }

  const int trials = 10;
  double scalar_total = 0;
  double simd_total = 0;

  // Run scalar trials
  for (int t = 0; t < trials; ++t) {
    auto start = std::chrono::high_resolution_clock::now();
    vector_add_scalar(A, B, C_scalar, N);
    auto end = std::chrono::high_resolution_clock::now();
    scalar_total += std::chrono::duration<double, std::milli>(end - start).count();
  }

  // Run SIMD trials
  for (int t = 0; t < trials; ++t) {
    auto start = std::chrono::high_resolution_clock::now();
    vector_add_avx2(A, B, C_simd, N);
    auto end = std::chrono::high_resolution_clock::now();
    simd_total += std::chrono::duration<double, std::milli>(end - start).count();
  }

  // Verify result correctness
  for (int i = 0; i < N; ++i) {
    if (fabs(C_scalar[i] - C_simd[i]) > 1e-5f) {
      std::cerr << "Mismatch at index " << i << ": " << C_scalar[i] << " vs " << C_simd[i] <<"\n";
      break;
    }
  }

  double scalar_avg = scalar_total / trials;
  double simd_avg = simd_total / trials;
  std::cout << "N = " << N << "\tScalar Avg: " << scalar_avg << " ms" << "\tSIMD Avg: " << simd_avg << " ms" << "\tSpeedup: " << (scalar_avg / simd_avg) << "x\n";

  _aligned_free(A);
  _aligned_free(B);
  _aligned_free(C_scalar);
  _aligned_free(C_simd);
}

int main() {
  std::cout << "Running SIMD tests...\n";
  for (int exp = 10; exp <= 20; exp += 2) {
    int N = 1 << exp;
    run_test(N);
  }
  std::cout << "Done.\n";
  return 0;
}
