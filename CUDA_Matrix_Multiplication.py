import cupy as cp
import numpy as np
import time

# Perform matrix multiplication
def run_matrix_multiplication(N):
  
  # Create random matrices
  A_cpu = np.random.rand(N, N).astype(np.float32)
  B_cpu = np.random.rand(N, N).astype(np.float32)
  A_gpu = cp.asarray(A_cpu)
  B_gpu = cp.asarray(B_cpu)
  
  # CPU timing
  start_cpu = time.perf_counter()
  C_cpu = np.matmul(A_cpu, B_cpu)
  end_cpu = time.perf_counter()
  cpu_time = (end_cpu - start_cpu) * 1000
  
  # GPU timing
  cp.cuda.Device(0).synchronize()
  start_gpu = time.perf_counter()
  C_gpu = cp.matmul(A_gpu, B_gpu)
  cp.cuda.Device(0).synchronize()
  end_gpu = time.perf_counter()
  gpu_time = (end_gpu - start_gpu) * 1000
  
  # Verify correctness
  C_gpu_cpu = cp.asnumpy(C_gpu)
  error = np.max(np.abs(C_cpu - C_gpu_cpu))
  
   # Print results
  print(f"N = {N} | CPU Time: {cpu_time:.3f} ms | GPU Time: {gpu_time:.3f} ms | Max Error: {error:.6f}")
  return N, cpu_time, gpu_time
  
# Run experiments
for exp in range(6, 11): # N=2^6=64 up to N=2^10=1024
  N = 2 ** exp
  run_matrix_multiplication(N)
