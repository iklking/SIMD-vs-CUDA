# A Comparative Study for Data-Parallel Algorithms in SIMD vs CUDA
+ Implemented vector addition and matrix multiplication in C++ to scalar and AVX2 SIMD parallelization based on execution time 
+ Developed equivalent Python implementations using NumPy (scalar) and CuPy (CUDA) in Google Colab to analyze GPU-based parallelization 
+ Explored CPU vs. GPU data-level parallelism, evaluating performance trade-offs and drawing conclusions from experimental results
## Files
### Report
Comparitive Study SIMD vs CUDA
+ Compares execution times of SIMD and CUDA against their scalar alternatives
+ Explores the idea of data parallelism
### SIMD
SIMD_Vector_Addition.cpp
+ Runs vector addition algorithms in scalar and AVX2 SIMD formats
+ Prints the execution times of each at varying input sizes
SIMD_Matrix_Multiplication.cpp
+ Runs matrix multiplication algorithms in scalar and AVX2 SIMD formats
+ Prints the execution times of each at varying input sizes
### CUDA
CUDA_Vector_Addition.py
+ Runs vector addition algorithms in scalar and CUDA formats
+ Uses NumPy and CuPy respectively
+ Prints the execution times of each at varying input sizes
CUDA_Matrix_Multiplication.py
+ Runs matrix multiplication algorithms in scalar and CUDA formats
+ Uses NumPy and CuPy respectively
+ Prints the execution times of each at varying input sizes
## Compiling
### SIMD_Vector_Addition.cpp
g++ -O2 -mavx2 -o vector_addition SIMD_Vector_Addition.cpp
### SIMD_Matrix_Multiplication.cpp
g++ -O2 -mavx2 -std=c++11 -o matrix_multiplication SIMD_Matrix_Multiplication.cpp
### CUDA_Vector_Addition.py
Executed in Google Colab
### CUDA_Matrix_Multiplication.py
Executed in Google Colab
